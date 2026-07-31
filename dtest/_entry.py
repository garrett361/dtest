# Modifed from DeepSpeed. Original header below.

################################################

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


import pytest

from dtest import DTest
from dtest._dtest import _closest_mark


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if item.cls and issubclass(item.cls, DTest):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


def pytest_configure(config):
    # All three resolve by pytest scope, closest first: the test, then its class, then its module.
    config.addinivalue_line(
        "markers",
        "world_size: set the world size(s) the test will run with. Must be in place by collection"
        " time, since it generates the test instances",
    )
    config.addinivalue_line("markers", "cpu: force cpu")
    config.addinivalue_line("markers", "gpu: force gpu")


def pytest_generate_tests(metafunc):
    """Generate separate test instances for each world_size value, if applicable."""
    # `world_size` is not a mark name we own exclusively, and this hook runs against every
    # test in the session, so leave other suites' identically named marks alone.
    if not (metafunc.cls and issubclass(metafunc.cls, DTest)):
        return

    mark = problem = None
    try:
        mark = _closest_mark(metafunc.definition, "world_size")
    except ValueError as e:
        problem = str(e)
    if mark is not None and len(mark.args) != 1:
        problem = (
            f"{metafunc.definition.nodeid}: the 'world_size' mark takes exactly one value, e.g."
            " @pytest.mark.world_size(2) or @pytest.mark.world_size([2, 4])"
        )
    if problem is not None:
        # Reported outside the `except` so pytest does not also render the original traceback, and
        # via `pytest.fail` because a bare raise mid-collection surfaces as an internal dtest one.
        # The ignore is a ty false positive: pytest casts `fail` to a Protocol whose `__call__` is
        # a bare TypeVar, which ty does not resolve back to the real signature.
        pytest.fail(problem, pytrace=False)  # ty: ignore
    if mark is None:
        return

    world_sizes = mark.args[0]
    # Ensure world_sizes is a list
    if not isinstance(world_sizes, (list, tuple)):
        world_sizes = [world_sizes]

    # parametrize only accepts names already in the fixture closure. Adding it lets tests
    # use the mark without also declaring a `world_size` arg they never read.
    if "world_size" not in metafunc.fixturenames:
        metafunc.fixturenames.append("world_size")

    # Parametrize the test with world_size values. Anything else that parametrizes `world_size` (a
    # parametrize mark, a fixture of that name, another plugin) already collides loudly as pytest's
    # own `duplicate parametrization of 'world_size'`, so there is nothing for us to add.
    metafunc.parametrize(
        "world_size", world_sizes, ids=[f"world_size={ws}" for ws in world_sizes]
    )
