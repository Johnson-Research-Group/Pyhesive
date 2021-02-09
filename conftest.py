#!/usr/bin/env python3
def pytest_addoption(parser):
    parser.addoption("--pyhesive-replace", action="store_true", help="replace output files during tests")

def pytest_report_header(config):
    msg = ["Command line arguments:"]
    msg.extend(config.invocation_params.args)
    return msg
