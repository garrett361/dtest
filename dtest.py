# Modifed from DeepSpeed. Original header below.

################################################

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
import multiprocessing as mp
import os
import socket
from functools import cache
import textwrap

import pytest
import torch
import torch.distributed as dist
from _pytest.fixtures import FixtureLookupError
from _pytest.outcomes import Skipped


@cache
def get_rank() -> int:
    return int(os.environ["RANK"])


@cache
def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


@cache
def get_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@cache
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"{get_device_type()}:{get_rank()}")
    return torch.device(f"{get_device_type()}:{get_rank()}")


@cache
def get_backend() -> str:
    return "nccl" if torch.cuda.is_available() else "gloo"


@cache
def get_num_gpus() -> int:
    if get_device_type() != "cuda":
        return 0
    return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def print_rank(s, *args, **kwargs):
    print(
        "\n".join(textwrap.wrap(s, initial_indent=f"[rank={get_rank()}] ")),
        *args,
        **kwargs,
    )


def print_rank0_only(s, *args, **kwargs):
    if not get_rank():
        print(
            "\n".join(textwrap.wrap(s, initial_indent=f"[rank={get_rank()}] ")),
            *args,
            **kwargs,
        )


# Worker timeout for tests that hang
DEEPSPEED_TEST_TIMEOUT = int(os.environ.get("DS_UNITTEST_TIMEOUT", "600"))


def get_master_port(base_port: int = 29500, port_range_size: int = 1000) -> str:
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

    world_size = 2
    backend = get_backend()
    requires_cuda_env = True
    exec_timeout = DEEPSPEED_TEST_TIMEOUT
    start_method = "spawn"

    def __call__(self, request):
        self._current_test = self._get_current_test_func(request)
        self._fixture_kwargs = self._get_fixture_kwargs(request, self._current_test)

        if self.requires_cuda_env and not torch.cuda.is_available():
            pytest.skip("only supported in accelerator environments.")

        # Catch world_size override pytest mark
        for mark in getattr(request.function, "pytestmark", []):
            if mark.name == "world_size":
                world_sizes = mark.args[0]
                break
        else:
            world_sizes = self._fixture_kwargs.get("world_size", self.world_size)

        if isinstance(world_sizes, int):
            world_sizes = [world_sizes]
        for ws in world_sizes:
            self._launch_procs(ws)

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
        if get_device_type() != "cpu" and get_num_gpus() < world_size:
            pytest.skip(
                f"Skipping test because not enough GPUs are available: {world_size} required, {get_num_gpus()} available"
            )

        # Set start method to `forkserver` (or `fork`)
        mp_context = mp.get_context(self.start_method)
        # mp.set_start_method("forkserver", force=True)
        # pool = mp.Pool(processes=world_size)
        master_port = get_master_port()

        # Run the test
        skip_msg_q = mp_context.Queue()
        args_list = [
            (rank, world_size, master_port, skip_msg_q) for rank in range(world_size)
        ]
        procs = [
            mp_context.Process(target=self._dist_run, args=args) for args in args_list
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        if not skip_msg_q.empty():
            # This assumed all skip messages are the same, it may be useful to
            # add a check here to assert all exit messages are equal
            pytest.skip(skip_msg_q.get())

    def _dist_run(
        self, rank: int, world_size: int, master_port: str, skip_msg_q: mp.Queue
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
            backend=self.backend,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()

        try:
            self._current_test(**self._fixture_kwargs)
        except BaseException as e:
            if isinstance(e, Skipped):
                if not get_rank():
                    print(f"Caught pytest.skip: {e}")
                    skip_msg_q.put(e.msg)
                raise e
            else:
                raise e
        finally:
            dist.destroy_process_group()
