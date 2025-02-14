# Modifed from DeepSpeed. Original header below.

################################################

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import datetime
import inspect
import multiprocessing as mp
import os
import socket
import time
import traceback
from typing import Union, Literal
import textwrap

import pytest
import torch
import torch.distributed as dist
from _pytest.fixtures import FixtureLookupError
from _pytest.outcomes import Skipped


def _get_master_port(base_port: int = 29500, port_range_size: int = 1000) -> str:
    # Select first open port in range
    port = base_port
    max_port = base_port + port_range_size
    sock = socket.socket()
    while port < max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return str(port)
        except OSError:
            port += 1
    raise IOError("no free ports")


class DTest:
    """
    Implementation for running pytest with distributed execution.

    Args:
        - default_world_size: int | "auto" = "auto" -- the number of processes to launch.

    Features:
        - able to call pytest.skip() inside tests
        - works with pytest fixtures, parametrize, mark, etc.
        - can contain multiple tests (each of which can be parametrized separately)
        - class methods can be fixtures (usable by tests in this class only)
        - world_size can be changed for individual tests using @pytest.mark.world_size(world_size)

    Usage:
        - class name must start with "Test"
        - must implement one or more test*(self, ...) methods

    Example:
        @pytest.fixture(params=[10,20])
        def val1(request):
            return request.param

        @pytest.mark.fast
        @pytest.mark.parametrize("val2", [30,40])
        class TestExample(DistributedTest):
            default_world_size = 2

            @pytest.fixture(params=[50,60])
            def val3(self, request):
                return request.param

            def test_1(self, val1, val2, str1="hello world"):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                assert all(val1, val2, str1)

            @pytest.mark.world_size(1)
            @pytest.mark.parametrize("val4", [70,80])
            def test_2(self, val1, val2, val3, val4):
                assert int(os.environ["WORLD_SIZE"]) == 1
                assert all(val1, val2, val3, val4)
    """

    default_world_size: Union[int, Literal["auto"]] = "auto"
    requires_cuda_env = True
    start_method = "spawn"
    _force_gpu = False
    _force_cpu = False
    _poll_sec = 1
    _init_timeout_sec = 60

    def __call__(self, request):
        self._current_test = self._get_current_test_func(request)
        self._fixture_kwargs = self._get_fixture_kwargs(request, self._current_test)

        if self.requires_cuda_env and not torch.cuda.is_available():
            pytest.skip("only supported in accelerator environments.")

        # Process DTest specific marks: {world_size, gpu, cpu}
        mark_dict = {
            mark.name: mark for mark in getattr(request.function, "pytestmark", [])
        }
        # Catch world_size override pytest mark
        if "world_size" in mark_dict:
            world_sizes = mark_dict["world_size"].args[0]
        else:
            world_sizes = self._fixture_kwargs.get(
                "world_size", self.default_world_size
            )

        # If world_size = "auto", try to read from CUDA_VISIBLE_DEVICES, otherwise default to 2
        if isinstance(world_sizes, str):
            if world_sizes != "auto":
                raise ValueError("The only valid string for world_size is 'auto'")
            world_sizes = self.num_gpus() or 2

        if isinstance(world_sizes, int):
            world_sizes = [world_sizes]

        if "cpu" in mark_dict and "gpu" in mark_dict:
            raise ValueError("Only one of 'cpu' or 'gpu' may be marked")
        if "cpu" in mark_dict:
            self._force_cpu = True
        if "gpu" in mark_dict:
            self._force_gpu = True

        try:
            for ws in world_sizes:
                self._launch_procs(ws)
        finally:
            self._force_gpu = False
            self._force_cpu = False

    def _get_current_test_func(self, request):
        # DistributedTest subclasses may have multiple test methods
        func_name = request.function.__name__
        return getattr(self, func_name)

    def _get_fixture_kwargs(self, request, func):
        if not request:
            return {}
        # Grab fixture / parametrize kwargs from pytest request object
        fixture_kwargs = {}
        params = inspect.getfullargspec(func).args
        params.remove("self")
        for p in params:
            try:
                fixture_kwargs[p] = request.getfixturevalue(p)
            except FixtureLookupError:
                pass  # test methods can have kwargs that are not fixtures
        return fixture_kwargs

    def _launch_procs(self, world_size):
        # Verify we have enough accelerator devices to run this test
        if self.device_type != "cpu" and self.num_gpus() < world_size:
            pytest.skip(
                f"Skipping test because not enough GPUs are available: {world_size} required, {self.num_gpus()} available"
            )

        mp_context = mp.get_context(self.start_method)
        master_port = _get_master_port()

        # Run the test
        ex_q = mp_context.Queue()
        skip_q = mp_context.Queue()
        args_list = [
            (rank, world_size, master_port, skip_q, ex_q) for rank in range(world_size)
        ]
        procs_dict = {
            rank: mp_context.Process(target=self._dist_run, args=args)
            for rank, args in enumerate(args_list)
        }
        for p in procs_dict.values():
            p.start()
        while procs_dict:
            if not skip_q.empty():
                for p in procs_dict.values():
                    p.terminate()
                pytest.skip(skip_q.get())

            if not ex_q.empty():
                rank, tb, e = ex_q.get()
                print(f"TRACEBACK from Rank {rank}")
                print(tb)
                for p in procs_dict.values():
                    p.terminate()
                raise e

            ranks_to_remove = []
            non_zero_exit_code_ranks = []
            for rank, p in procs_dict.items():
                if not p.is_alive():
                    if p.exitcode != 0:
                        non_zero_exit_code_ranks.append((rank, p.exitcode))
                    ranks_to_remove.append(rank)
            for rank in ranks_to_remove:
                del procs_dict[rank]

            if non_zero_exit_code_ranks:
                for p in procs_dict.values():
                    p.terminate()
                raise RuntimeError(
                    f"Found non-zero exit codes from these rank, exit code pairs: "
                    f"{non_zero_exit_code_ranks}"
                )

            time.sleep(self._poll_sec)

    def _dist_run(
        self,
        rank: int,
        world_size: int,
        master_port: str,
        skip_q: mp.Queue,
        ex_q: mp.Queue,
    ):
        """Initialize deepspeed.comm and execute the user function."""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # turn off NCCL logging if set
        os.environ.pop("NCCL_DEBUG", None)

        if self.device_type == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
            device_id=self.device if self.backend == "nccl" else None,
            timeout=datetime.timedelta(seconds=self._init_timeout_sec),
        )
        dist.barrier()

        try:
            self._current_test(**self._fixture_kwargs)
        except BaseException as e:
            if isinstance(e, Skipped):
                skip_q.put(e.msg)
            else:
                tb = traceback.format_exc()
                ex_q.put((rank, tb, e))
                raise e
        finally:
            dist.barrier()
            dist.destroy_process_group()

    @property
    def rank(self) -> int:
        return int(os.getenv("RANK", 0))

    @property
    def world_size(self) -> int:
        return int(os.getenv("WORLD_SIZE", 1))

    @property
    def device_type(self) -> str:
        if self._force_gpu:
            return "cuda"
        elif self._force_cpu or not self.requires_cuda_env:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> torch.device:
        return torch.device(f"{self.device_type}:{self.rank}")

    @property
    def backend(self) -> str:
        if self._force_gpu:
            return "nccl"
        elif self._force_cpu or not self.requires_cuda_env:
            return "gloo"

        return "nccl" if torch.cuda.is_available() else "gloo"

    def num_gpus(self) -> int:
        if self.device_type != "cuda":
            return 0
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    def print_rank(self, s, *args, **kwargs):
        print(
            "\n".join(textwrap.wrap(s, initial_indent=f"[rank={self.rank}] ")),
            *args,
            **kwargs,
        )

    def print_rank0_only(self, s, *args, **kwargs):
        if not self.rank:
            print(
                "\n".join(textwrap.wrap(s, initial_indent=f"[rank={self.rank}] ")),
                *args,
                **kwargs,
            )
