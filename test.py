import pytest
from dtest import DTest
import torch
import torch.distributed as dist


class TestDTest(DTest):
    requires_cuda_env = False

    def test_basic(self) -> None:
        print(f"{self.get_rank()=}")

    def test_all_reduce(self) -> None:
        t = torch.arange(self.get_world_size(), device=self.get_device())
        dist.all_reduce(t)
        self.print_rank(f"{t=}")

    def test_skip(self) -> None:
        pytest.skip("I should be skipped")

    def test_fail(self) -> None:
        self.print_rank0_only("I should fail")
        assert False

    @pytest.mark.world_size([2, 3, 4])
    def test_world_sizes(self) -> None:
        self.print_rank0_only(f"{self.get_world_size()=}")

    @pytest.mark.parametrize("n", (2, 3, 4))
    def test_parametrize(self, n) -> None:
        self.print_rank0_only(f"{n=}")


def test_regular():
    print("In regular test")
    assert True
