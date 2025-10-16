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
    def test_non_default_world_size(self, world_size: int) -> None:
        self.print_rank0_only(f"{self.world_size=}")

    @pytest.mark.world_size([2, 3, 4])
    def test_multiple_world_sizes(self, world_size: int) -> None:
        self.print_rank0_only(f"{self.world_size=}")

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

    @pytest.mark.world_size(4)
    def test_shared_tmp_file(self, world_size: int) -> None:
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


class TestOtherDefaultWorldSizeDTest(DTest):
    requires_cuda_env = False  # Just for running on CPU
    default_world_size = 7

    def test_default_world_size(self) -> None:
        self.print_rank0_only(f"{self.world_size=}")


def test_regular():
    print("In regular test")
    assert True
