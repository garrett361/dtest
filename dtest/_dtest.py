# Modifed from DeepSpeed. Original header below.

################################################

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import datetime
import inspect
import os
import pathlib
import socket
import tempfile
import textwrap
import time
from contextlib import contextmanager
from random import randint
from typing import Any, Callable, Generator, Literal, Optional, Union

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _pytest.fixtures import FixtureLookupError
from _pytest.outcomes import Skipped
from torch.distributed.elastic.multiprocessing.api import (
    DefaultLogsSpecs,
    MultiprocessContext,
)
from torch.distributed.elastic.multiprocessing.errors import record


def _print_dict_flattened(d, prefix: str = "") -> None:
    if not isinstance(d, dict):
        print(d)
        return
    for k, v in d.items():
        if isinstance(v, dict):
            _print_dict_flattened(v, k if not prefix else prefix + f".{k}")
        else:
            printed_prefix = (k if not prefix else prefix + f".{k}").upper() + ": "
            print(printed_prefix, v, "\n")


def _get_master_port(
    base_port: int = 29500, port_range_size: int = 1000, max_tries: int = 10
) -> str:
    max_port = base_port + port_range_size
    sock = socket.socket()
    tries = 0
    while tries < max_tries:
        try:
            port = randint(base_port, max_port)
            sock.bind(("", port))
            sock.close()
            return str(port)
        except OSError:
            tries += 1
    raise IOError("no free ports")


class DTestFailedError(Exception):
    def __init__(self, message="DTest Failed"):
        self.message = message
        super().__init__(self.message)


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
    start_method: str = "spawn"
    no_nccl_debug: bool = True
    _force_gpu: bool = False
    _force_cpu: bool = False
    _poll_sec: int = 1
    _init_timeout_sec: int = 30
    _seed: Optional[int] = 42

    def __call__(self, request):
        test = self._get_current_test_func(request)
        test_kwargs = self._get_fixture_kwargs(request, test)

        # Process DTest specific marks: {world_size, gpu, cpu}
        mark_dict = {
            mark.name: mark for mark in getattr(request.function, "pytestmark", [])
        }
        # Catch world_size override pytest mark
        if (
            hasattr(request.node, "callspec")
            and "world_size" in request.node.callspec.params
        ):
            world_size = request.node.callspec.params["world_size"]
        elif "world_size" in mark_dict:
            world_size = mark_dict["world_size"].args[0]
        else:
            world_size = test_kwargs.get("world_size", self.default_world_size)

        # If world_size = "auto", try to read from CUDA_VISIBLE_DEVICES, otherwise default to 2
        if isinstance(world_size, str):
            if world_size != "auto":
                raise ValueError("The only valid string for world_size is 'auto'")
            world_size = self.num_gpus() or 2

        if "cpu" in mark_dict and "gpu" in mark_dict:
            raise ValueError("Only one of 'cpu' or 'gpu' may be marked")
        if "cpu" in mark_dict:
            self._force_cpu = True
        if "gpu" in mark_dict:
            self._force_gpu = True

        try:
            self.run(test, test_kwargs, world_size)
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

    def run(
        self, test: Callable[..., None], test_kwargs: dict[Any, Any], world_size: int
    ):
        # Verify we have enough accelerator devices to run this test
        if self._force_gpu and self.device_type == "cpu":

            pytest.skip(
                f"{self.__class__.__name__}:{test.__name__} requires GPUs, but none available."
            )
        if self.device_type == "gpu" and self.num_gpus() < world_size:
            pytest.skip(
                f"Insufficient GPUs available for {self.__class__.__name__}:{test.__name__}:"
                f" {world_size} required, {self.num_gpus()} available."
            )

        mp_context = mp.get_context(self.start_method)
        master_port = _get_master_port()

        # Run the test
        skip_q = mp_context.Queue()
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file_name = file.name
            args = {
                r: (test, test_kwargs, skip_q, file_name) for r in range(world_size)
            }
            envs = {}
            for local_rank in range(world_size):
                worker_env = {
                    "LOCAL_RANK": str(local_rank),
                    "RANK": str(local_rank),
                    "WORLD_SIZE": str(world_size),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": master_port,
                }
                envs[local_rank] = worker_env
            log_line_prefixes = {r: f"[rank {r}]" for r in range(world_size)}
            context = MultiprocessContext(
                name="dtest",
                entrypoint=self._dist_run,
                args=args,
                envs=envs,
                start_method=self.start_method,
                logs_specs=DefaultLogsSpecs(),
                log_line_prefixes=log_line_prefixes,
            )
            try:
                context.start()
                while True:
                    if not skip_q.empty():
                        # TODO: @goon -  KILL PROCS
                        context.close()
                        pytest.skip(skip_q.get())

                    result = context.wait(0)
                    if result:
                        if result.is_failed():
                            for local_rank, proc_failure in result.failures.items():
                                print(
                                    f"FAILURE on {local_rank=}:\n",
                                )
                                _print_dict_flattened(proc_failure.message)
                            raise DTestFailedError(proc_failure.message["message"])
                        return
                    time.sleep(self._poll_sec)

            except Exception:
                context.close()
                raise

    # NOTE: @goon - important to have this record here to successfully capture some types of NCCL
    # errors, it seems.
    @record
    def _dist_run(
        self,
        test: Callable[..., None],
        test_kwargs: dict[Any, Any],
        skip_q: mp.Queue,
        file_name: str,
    ):
        rank = local_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # turn off NCCL logging if set
        if self.no_nccl_debug:
            os.environ.pop("NCCL_DEBUG", None)

        if self.device_type == "cuda":
            torch.cuda.set_device(local_rank)
        # For unknown reasons, setting some subset of {rank, world_size, and device_id}
        # can cause dist.{send,receive} calls to fail, so we omit them.
        if self._seed is not None:
            torch.manual_seed(self._seed)
        store = dist.FileStore(file_name, world_size)
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
            # device_id=self.device if self.backend == "nccl" else None,
            timeout=datetime.timedelta(seconds=self._init_timeout_sec),
            store=store,
        )
        dist.barrier()

        try:
            test(**test_kwargs)
        except Skipped as e:
            skip_q.put(e.msg)
        finally:
            # NOTE: @goon - Previously had a `dist.barrier` call here to sync procs, and this was a
            # BAD idea, since the barrier is **always** called in this pattern. When a test would
            # fail, the resulting stack trace would then be from the failed barrier call, rather
            # than from the actual underlying failure.
            dist.destroy_process_group()

    @property
    def rank(self) -> int:
        return int(os.getenv("RANK", 0))

    @property
    def world_size(self) -> int:
        return int(os.getenv("WORLD_SIZE", 1))

    @property
    def device_type(self) -> str:
        if not torch.cuda.is_available() or self._force_cpu:
            return "cpu"
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device(f"{self.device_type}:{self.rank}")

    @property
    def backend(self) -> str:
        if not torch.cuda.is_available() or self._force_cpu:
            return "gloo"
        return "nccl"

    def num_gpus(self) -> int:
        if self.device_type != "cuda":
            return 0
        return torch.cuda.device_count()

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

    @contextmanager
    def temp_dir(self) -> Generator[pathlib.Path, None, None]:
        """
        Create a shared temp dir for writing to.
        """
        if not self.rank:
            temp_dir = tempfile.TemporaryDirectory()
            temp_dir_name = temp_dir.name
        else:
            temp_dir_name = None
        temp_dir_name_list = [temp_dir_name]
        dist.broadcast_object_list(temp_dir_name_list, src=0)
        yield pathlib.Path(temp_dir_name_list[0])
        dist.barrier()
