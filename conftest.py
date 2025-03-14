# Modifed from DeepSpeed. Original header below.

################################################

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# tests directory-specific settings - this file is run automatically by pytest before any tests are run

import pytest
from dtest import DTest


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if item.cls and issubclass(item.cls, DTest):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


def pytest_configure(config):
    config.addinivalue_line("markers", "world_size(s)")
    config.addinivalue_line("markers", "cpu")
    config.addinivalue_line("markers", "gpu")
