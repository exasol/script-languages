#!/usr/bin/env python3


from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.available_python_packages_utils import run_python_package_import_test

AVAILABLE_PACKAGES_SCHEMA = "available_packages"


class AvailablePython3Packages(udf.TestCase):
    def setUp(self): 
        self.query(f'create schema {AVAILABLE_PACKAGES_SCHEMA}', ignore_errors=True)

    data = [
            ("sagemaker",),
        ]

    @useData(data)
    def test_package_import(self, pkg, fail=False, alternative=None):
        run_python_package_import_test(self, AVAILABLE_PACKAGES_SCHEMA, "PYTHON3", pkg, fail, alternative)

if __name__ == '__main__':
    udf.main()
