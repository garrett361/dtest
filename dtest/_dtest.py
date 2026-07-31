# Modifed from DeepSpeed. Original header below.

################################################

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import datetime
import inspect
import os
import pathlib
import signal
import socket
import tempfile
import textwrap
import time
from contextlib import contextmanager, suppress
from random import randint
from typing import Any, Callable, Generator, Literal, Optional, Union

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.api import (
    DefaultLogsSpecs,
    MultiprocessContext,
    SignalException,
)
from torch.distributed.elastic.multiprocessing.errors import record


def _format_failure(local_rank: int, message: Union[str, dict]) -> str:
    if isinstance(message, str):
        body = message
    else:
        body = message.get("extraInfo", {}).get("py_callstack", "") or message.get(
            "message", "unknown error"
        )
    prefix = f"[rank {local_rank}] "
    return textwrap.indent(body, prefix)


def _closest_mark(node, *names: str):
    """The closest mark among `names`, or None.

    Scope resolution is pytest's own: the test, then its class, then its module. Pytest folds a
    class together with its base classes into a single scope, so rather than invent an inheritance
    order that disagrees with how `skipif` and every other mark resolve, more than one applicable
    mark at one scope is an error.
    """
    scoped_marks = [
        (scope, mark)
        for scope, mark in node.iter_markers_with_node()
        if mark.name in names
    ]
    if not scoped_marks:
        return None
    closest_scope, closest_mark = scoped_marks[0]
    # pytest concatenates a class's MRO marks without deduping, so a base and a subclass repeating
    # the same mark are not in conflict. Compare by value; Mark is a frozen dataclass.
    marks_at_scope = [mark for scope, mark in scoped_marks if scope is closest_scope]
    distinct_marks = [
        mark for i, mark in enumerate(marks_at_scope) if mark not in marks_at_scope[:i]
    ]
    if len(distinct_marks) > 1:
        mark_names = ", ".join(sorted({repr(mark.name) for mark in distinct_marks}))
        raise ValueError(
            f"{closest_scope.nodeid}: conflicting {mark_names} marks at one scope, keep a single"
            " mark. A class and its base classes count as one scope, as do a test and the"
            " pytest.param marks of its parameters."
        )
    return closest_mark


def _resolve_device_marks(instance: "DTest", node) -> None:
    """Record the closest cpu/gpu mark on `instance`."""
    device = _closest_mark(node, "cpu", "gpu")
    instance._force_cpu = device is not None and device.name == "cpu"
    instance._force_gpu = device is not None and device.name == "gpu"


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
    raise IOError(
        f"No free ports in range [{base_port}, {max_port}] after {max_tries} attempts"
    )


class DTest:
    """
    Implementation for running pytest with distributed execution.

    Args:
        - default_world_size: int | "auto" = "auto" -- the number of processes to launch.

    Features:
        - able to call pytest.skip() inside tests
        - works with pytest fixtures, parametrize, mark, etc.
        - can contain multiple tests (each of which can be parametrized separately)
        - class methods can be fixtures (usable by tests in this class only), but they run in the
          parent rather than in a rank: `rank`, `world_size`, `device` and anything built on them
          raise there, and `device_type`, `backend` and `num_gpus` raise unless the fixture is
          function-scoped
        - world_size can be changed for individual tests using @pytest.mark.world_size(world_size)
        - the world_size, cpu and gpu marks resolve like any other pytest mark: the test's own
          mark wins, then its class's, then its module's

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
    _poll_sec: int = 1
    _init_timeout_sec: int = 30
    # `None` rather than False until the cpu/gpu marks are resolved, so a read that cannot see them
    # raises instead of reporting the wrong device. Only `_force_cpu` needs it; `_force_gpu` is read
    # only by `run`, after the assignment.
    _force_cpu: Optional[bool] = None
    _force_gpu: Optional[bool] = None
    # True only on the copy of this instance that `_dist_run` unpickles inside a spawned rank.
    _is_worker: bool = False
    _seed: Optional[int] = 42

    def __call__(self, request):
        test = self._get_current_test_func(request)
        test_kwargs = self._get_fixture_kwargs(request, test)

        _resolve_device_marks(self, request.node)

        # Resolve world_size (after device marks so num_gpus respects _force_cpu)
        if (
            hasattr(request.node, "callspec")
            and "world_size" in request.node.callspec.params
        ):
            world_size = request.node.callspec.params["world_size"]
        else:
            # A collection-time mark always parametrizes, so one still visible here arrived too late
            # to generate test instances. Better to say so than to silently use the wrong size.
            if _closest_mark(request.node, "world_size") is not None:
                raise ValueError(
                    f"{self.__class__.__name__}:{test.__name__}: 'world_size' marks must be applied"
                    " to the test, its class, or its module, not via add_marker or pytest.param"
                )
            world_size = test_kwargs.get("world_size", self.default_world_size)

        if isinstance(world_size, str):
            if world_size != "auto":
                raise ValueError(
                    f"{self.__class__.__name__}:{test.__name__}: world_size must be int or 'auto', got {world_size!r}"
                )
            world_size = self.num_gpus or 2

        self.run(test, test_kwargs, world_size)

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
            except pytest.FixtureLookupError:
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
        if self.device_type == "cuda" and self.num_gpus < world_size:
            pytest.skip(
                f"Insufficient GPUs available for {self.__class__.__name__}:{test.__name__}:"
                f" {world_size} required, {self.num_gpus} available."
            )

        mp_context = mp.get_context(self.start_method)
        master_port = _get_master_port()

        # Run the test
        skip_q = mp_context.Queue()
        # NOTE: @goon - `delete=False` is load-bearing: `dist.FileStore` unlinks this file once
        # every rank has destroyed its store, so letting the context manager also unlink it raises
        # FileNotFoundError. Clean up by hand instead, since FileStore never gets there when a rank
        # dies early or the run is interrupted.
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file_name = file.name
        try:
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
                            if any(
                                pf.exitcode == -signal.SIGINT
                                for pf in result.failures.values()
                            ):
                                context.close()
                                raise KeyboardInterrupt
                            messages = []
                            for local_rank, proc_failure in result.failures.items():
                                messages.append(
                                    _format_failure(local_rank, proc_failure.message)
                                )
                            pytest.fail("\n".join(messages), pytrace=False)
                        return
                    time.sleep(self._poll_sec)

            except SignalException:
                context.close()
                raise KeyboardInterrupt
            except BaseException:
                context.close()
                raise
        finally:
            with suppress(FileNotFoundError):
                os.unlink(file_name)

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
        self._is_worker = True
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
        except pytest.skip.Exception as e:
            skip_q.put(e.msg)
            return  # parent detects via skip_q and closes context
        except BaseException:
            # Don't call destroy_process_group() on failure: other ranks may be
            # blocked in NCCL collectives, and destroy attempts a graceful
            # shutdown that waits for them — deadlocking until collective timeout
            # and hiding the real error. Covers KeyboardInterrupt/SystemExit too.
            # The OS and CUDA driver reclaim all resources on process exit.
            raise
        else:
            dist.destroy_process_group()

    @property
    def rank(self) -> int:
        return self._worker_env_int("RANK")

    @property
    def world_size(self) -> int:
        return self._worker_env_int("WORLD_SIZE")

    def _worker_env_int(self, var: str) -> int:
        # Gated on `_is_worker` rather than on the variable itself, which the parent may have
        # inherited.
        if not self._is_worker:
            raise RuntimeError(
                f"{self.__class__.__name__}.{var.lower()} is only meaningful in a spawned rank,"
                " so it is unavailable in a fixture or anywhere else in the parent."
            )
        return int(os.environ[var])

    def _check_device_marks_resolved(self) -> None:
        # Names no property: `device` and `num_gpus` arrive through `device_type`, and naming that
        # would point at code the caller did not write.
        if self._force_cpu is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: the 'cpu' and 'gpu' marks are not resolved on this"
                " instance. They are per test, so a class-scoped fixture cannot see them; make the"
                " fixture function-scoped, or read the device in the test body."
            )

    @property
    def device_type(self) -> str:
        # Before the cuda call, so an unresolved read raises whether or not GPUs are present.
        self._check_device_marks_resolved()
        if self._force_cpu or not torch.cuda.is_available():
            return "cpu"
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device(f"{self.device_type}:{self.rank}")

    @property
    def backend(self) -> str:
        self._check_device_marks_resolved()
        if self._force_cpu or not torch.cuda.is_available():
            return "gloo"
        return "nccl"

    @property
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
            td = tempfile.TemporaryDirectory()
            temp_dir_name = td.name
        else:
            td = None
            temp_dir_name = None
        temp_dir_name_list = [temp_dir_name]
        dist.broadcast_object_list(temp_dir_name_list, src=0)
        try:
            yield pathlib.Path(temp_dir_name_list[0])
        finally:
            dist.barrier()
            if td is not None:
                td.cleanup()
