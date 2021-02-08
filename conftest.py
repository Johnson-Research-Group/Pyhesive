#!/usr/bin/env python3
def pytest_addoption(parser):
    parser.addoption("--replace", action="store_true", help="replace output files during tests")
