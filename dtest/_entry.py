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
    config.addinivalue_line("markers", "world_size: set the world size(s) the test will run with")
    config.addinivalue_line("markers", "cpu: force cpu")
    config.addinivalue_line("markers", "gpu: force gpu")
