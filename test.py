import pytest
from dtest import (
    DTest,
    get_rank,
    get_device,
    get_world_size,
    print_rank,
    print_rank0_only,
)
import torch
import torch.distributed as dist


class TestDTest(DTest):
    requires_cuda_env = False

    def test_basic(self) -> None:
        print(f"{get_rank()=}")

    def test_all_reduce(self) -> None:
        t = torch.arange(get_world_size(), device=get_device())
        dist.all_reduce(t)
        print_rank(f"{t=}")

    def test_skip(self) -> None:
        pytest.skip("I should be skipped")

    @pytest.mark.world_size([2, 3, 4])
    def test_world_sizes(self) -> None:
        print_rank0_only(f"{get_world_size()=}")

    @pytest.mark.parametrize("n", (2, 3, 4))
    def test_parametrize(self, n) -> None:
        print_rank0_only(f"{n=}")


def test_regular():
    print("In regular test")
    assert True
