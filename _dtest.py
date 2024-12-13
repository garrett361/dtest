# Modifed from DeepSpeed. Original header below.

################################################

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
import multiprocessing as mp
import os
import socket
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

    There are 2 parameters that can be modified:
        - world_size: Union[int,List[int]] = 2 -- the number of processes to launch
        - backend: Literal['nccl','mpi','gloo'] = 'nccl' -- which backend to use

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
            world_size = 2

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

    world_size: Union[int, Literal["auto"]] = "auto"
    backend = None
    requires_cuda_env = True
    start_method = "spawn"
    _force_gpu = False
    _force_cpu = False

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
            world_sizes = self._fixture_kwargs.get("world_size", self.world_size)

        # If world_size = "auto", try to read from CUDA_VISIBLE_DEVICES, otherwise default to 2
        if isinstance(world_sizes, str):
            if world_sizes != "auto":
                raise ValueError("The only valid string for world_size is 'auto'")
            world_sizes = self.get_num_gpus() or 2

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
        if self.get_device_type() != "cpu" and self.get_num_gpus() < world_size:
            pytest.skip(
                f"Skipping test because not enough GPUs are available: {world_size} required, {self.get_num_gpus()} available"
            )

        mp_context = mp.get_context(self.start_method)
        master_port = _get_master_port()

        # Run the test
        ex_q = mp_context.Queue()
        skip_q = mp_context.Queue()
        args_list = [
            (rank, world_size, master_port, skip_q, ex_q) for rank in range(world_size)
        ]
        procs = [
            mp_context.Process(target=self._dist_run, args=args) for args in args_list
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        if not skip_q.empty():
            pytest.skip(skip_q.get())
        if not ex_q.empty():
            print(f"Found exception: {ex_q.get()}")
            raise

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

        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        dist.init_process_group(
            backend=self.get_backend(),
            rank=rank,
            world_size=world_size,
            device_id=self.get_device() if self.get_backend() == "nccl" else None,
        )
        dist.barrier()

        try:
            self._current_test(**self._fixture_kwargs)
        except BaseException as e:
            if isinstance(e, Skipped):
                skip_q.put(e.msg)
            else:
                ex_q.put(str(e))
                raise e
        finally:
            dist.destroy_process_group()

    def get_rank(self) -> int:
        return int(os.getenv("RANK", 0))

    def get_world_size(self) -> int:
        return int(os.getenv("WORLD_SIZE", 1))

    def get_device_type(self) -> str:
        if self._force_gpu:
            return "cuda"
        elif self._force_cpu:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"{self.get_device_type()}:{self.get_rank()}")
        return torch.device(f"{self.get_device_type()}:{self.get_rank()}")

    def get_backend(self) -> str:
        if self._force_gpu:
            return "nccl"
        elif self._force_cpu:
            return "gloo"

        return "nccl" if torch.cuda.is_available() else "gloo"

    def get_num_gpus(self) -> int:
        if self.get_device_type() != "cuda":
            return 0
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    def print_rank(self, s, *args, **kwargs):
        print(
            "\n".join(textwrap.wrap(s, initial_indent=f"[rank={self.get_rank()}] ")),
            *args,
            **kwargs,
        )

    def print_rank0_only(self, s, *args, **kwargs):
        if not self.get_rank():
            print(
                "\n".join(
                    textwrap.wrap(s, initial_indent=f"[rank={self.get_rank()}] ")
                ),
                *args,
                **kwargs,
            )
