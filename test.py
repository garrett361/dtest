import pytest
import torch
import torch.distributed as dist

from dtest import DTest


def fn_for_traceback_testing():
    print("I should fail")
    assert False, "asserting False"


class TestDTest(DTest):
    requires_cuda_env = False

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

    def test_default_world_size(self) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.world_size(5)
    def test_non_default_world_size(self, world_size: int) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.world_size([2, 3, 4])
    def test_multiple_world_sizes(self, world_size: int) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.parametrize("n", (2, 3, 4))
    def test_parametrize(self, n) -> None:
        self.print_rank0_only(f"{n=}")


def test_regular():
    print("In regular test")
    assert True
