# Modifed from DeepSpeed. Original header below.

################################################

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


import pytest

from dtest import DTest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if item.cls and issubclass(item.cls, DTest):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "world_size: set the world size(s) the test will run with"
    )
    config.addinivalue_line("markers", "cpu: force cpu")
    config.addinivalue_line("markers", "gpu: force gpu")


def pytest_generate_tests(metafunc):
    """Generate separate test instances for each world_size value, if applicable."""
    # `world_size` is not a mark name we own exclusively, and this hook runs against every
    # test in the session, so leave other suites' identically named marks alone.
    if not (metafunc.cls and issubclass(metafunc.cls, DTest)):
        return

    mark_dict = {
        mark.name: mark for mark in getattr(metafunc.function, "pytestmark", [])
    }

    if "world_size" in mark_dict:
        world_sizes = mark_dict["world_size"].args[0]
        # Ensure world_sizes is a list
        if not isinstance(world_sizes, (list, tuple)):
            world_sizes = [world_sizes]

        # parametrize only accepts names already in the fixture closure. Adding it lets tests
        # use the mark without also declaring a `world_size` arg they never read.
        if "world_size" not in metafunc.fixturenames:
            metafunc.fixturenames.append("world_size")

        # Parametrize the test with world_size values
        metafunc.parametrize(
            "world_size", world_sizes, ids=[f"world_size={ws}" for ws in world_sizes]
        )
