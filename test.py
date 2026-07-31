import pytest
import torch
import torch.distributed as dist

from dtest import DTest


def fn_for_traceback_testing():
    print("I should fail")
    assert False, "asserting False"


class TestDTest(DTest):
    requires_cuda_env = False  # Just for running on CPU

    def test_basic(self) -> None:
        print(f"{self.rank=}")

    def test_all_reduce(self) -> None:
        t = torch.arange(self.world_size, device=self.device)
        dist.all_reduce(t)
        self.print_rank(f"{t=}")

    def test_skip(self) -> None:
        pytest.skip("I should be skipped")

    def test_fail(self) -> None:
        fn_for_traceback_testing()

    def test_nice_printing(self) -> None:
        self.print_rank(f"Hi from {self.rank=}")

    def test_default_world_size(self) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.world_size(5)
    def test_non_default_world_size(self) -> None:
        assert self.world_size == 5
        self.print_rank0_only(f"{self.world_size=}")

    # Values exclude the CPU default of 2, so an ignored mark fails rather than passing.
    @pytest.mark.world_size([3, 4, 5])
    def test_multiple_world_sizes(self) -> None:
        assert self.world_size in (3, 4, 5)
        self.print_rank0_only(f"{self.world_size=}")

    # The world_size arg is optional, and was previously required. Declaring it is the only
    # way to pin each generated instance to its own marked value.
    @pytest.mark.world_size([2, 3, 4])
    def test_world_size_arg_bc_check(self, world_size: int) -> None:
        assert world_size == self.world_size
        self.print_rank0_only(f"{world_size=}")

    @pytest.mark.parametrize("n", (2, 3, 4))
    def test_parametrize(self, n) -> None:
        self.print_rank0_only(f"{n=}")

    @pytest.mark.cpu
    def test_force_cpu(self) -> None:
        self.print_rank0_only(f"{self.device_type=}")
        self.print_rank0_only(f"{self.num_gpus=}")
        self.print_rank0_only(f"{self.backend=}")

    @pytest.mark.gpu
    def test_force_gpu(self) -> None:
        self.print_rank0_only(f"{self.device_type=}")
        self.print_rank0_only(f"{self.num_gpus=}")
        self.print_rank0_only(f"{self.backend=}")

    def test_root_error_visible_on_hang(self) -> None:
        """Rank 0 fails; rank 1 is stuck in a collective rank 0 won't enter."""
        if dist.get_rank() == 0:
            assert False, "intentional rank 0 failure"
        else:
            dist.barrier()

    @pytest.mark.world_size(4)
    def test_shared_tmp_file(self) -> None:
        filename = "hello.txt"
        with self.temp_dir() as tmp_dir:
            shared_file = tmp_dir / filename
            if self.rank == 0:
                with open(shared_file, "w") as f:
                    f.write(f"Hello from {self.rank=}")
                dist.barrier()
            else:
                dist.barrier()
                with open(shared_file, "r") as f:
                    self.print_rank(f.read())


@pytest.mark.cpu
@pytest.mark.world_size(3)
class TestClassLevelMarks(DTest):
    """Class-level marks apply to every test in the class."""

    def test_class_world_size(self) -> None:
        assert self.world_size == 3

    def test_class_device(self) -> None:
        assert self.device_type == "cpu"
        assert self.backend == "gloo"


@pytest.mark.gpu
class TestMarkPrecedence(DTest):
    """The closest mark wins."""

    @pytest.mark.cpu
    def test_method_mark_beats_class_mark(self) -> None:
        assert self.device_type == "cpu"

    def test_class_mark_applies(self) -> None:
        # Skips without GPUs.
        assert self.device_type == "cuda"


@pytest.mark.cpu
@pytest.mark.world_size(3)
class _MarkedBase(DTest):
    """Not collected: the name does not start with `Test`."""


class TestInheritedMarks(_MarkedBase):
    """Marks on a base class apply to its subclasses."""

    def test_inherited_marks(self) -> None:
        assert self.world_size == 3
        assert self.device_type == "cpu"


@pytest.mark.cpu
@pytest.mark.world_size(3)
class TestRepeatedMarks(_MarkedBase):
    """Repeating a base class's mark is fine; only a differing value is a conflict."""

    def test_repeated_marks(self) -> None:
        assert self.world_size == 3


class TestOtherDefaultWorldSizeDTest(DTest):
    requires_cuda_env = False  # Just for running on CPU
    default_world_size = 7

    def test_default_world_size(self) -> None:
        self.print_rank0_only(f"{self.world_size=}")


def test_regular():
    print("In regular test")
    assert True


@pytest.mark.parametrize("prop", ("device_type", "backend", "device", "num_gpus"))
def test_device_props_raise_before_marks_resolve(prop: str) -> None:
    """`device` and `num_gpus` are only guarded through `device_type`, so cover all four."""
    with pytest.raises(RuntimeError, match="marks are not resolved"):
        getattr(DTest(), prop)


@pytest.mark.parametrize("prop", ("rank", "world_size"))
def test_rank_props_ignore_an_inherited_env(prop: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A RANK the parent happened to inherit is not this process's, so it must not be read."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="only meaningful in a spawned rank"):
        getattr(DTest(), prop)


@pytest.mark.gpu
@pytest.mark.world_size(2)
class TestFixtureDeviceScopes(DTest):
    """Only function-scoped fixtures can see the device marks."""

    @pytest.fixture
    def device_function_scoped(self) -> str:
        # The class is marked gpu and this method overrides it, so the flags prove the mark was
        # resolved per test. Asserting `device_type` alone would not: it is "cpu" either way here.
        assert (self._force_cpu, self._force_gpu) == (True, False)
        return self.device_type

    @pytest.fixture(scope="class")
    def device_class_scoped(self) -> str:
        # Bound to a collection-time instance shared by every test in the class, so a per-test
        # mark has no single right answer to give it.
        with pytest.raises(RuntimeError, match="marks are not resolved"):
            _ = self.device_type
        return "raised"

    @pytest.fixture
    def rank_in_fixture(self) -> str:
        # Unreachable by any mark: the parent is not a rank.
        for prop in ("rank", "world_size", "device"):
            with pytest.raises(RuntimeError, match="only meaningful in a spawned rank"):
                getattr(self, prop)
        return "raised"

    @pytest.mark.cpu
    def test_device_by_fixture_scope(
        self, device_function_scoped: str, device_class_scoped: str, rank_in_fixture: str
    ) -> None:
        assert device_function_scoped == "cpu"
        assert device_class_scoped == "raised"
        assert rank_in_fixture == "raised"
        # All of them resolve in the body, the only place they are meaningful.
        assert self.device_type == "cpu"
        assert self.world_size == 2
        assert self.rank in range(2)
