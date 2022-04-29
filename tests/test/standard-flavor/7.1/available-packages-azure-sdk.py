#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.available_python_packages_utils import run_python_package_import_test


class AvailablePython3Packages(udf.TestCase):

    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True)

    data = [
            ("azure.batch",), 
            ("azure.cosmos",), 
            ("azure.eventgrid",),
            ("azure.eventhub",),
            ("azure.eventhub.extensions.checkpointstoreblob",),
            ("azure.eventhub.extensions.checkpointstoreblobaio",),
            ("azure.identity",),
            ("azure.keyvault",),
            ("azure.keyvault.certificates",),
            ("azure.keyvault.keys",),
            ("azure.keyvault.secrets",),
            ("azure.kusto.data",),
            ("azure.loganalytics",),
            ("azure.servicebus",),
            ("azure.storage.blob",),
            ("azure.storage.filedatalake",),
            ("azure.storage.fileshare",),
            ("azure.storage.queue",),
        ]

    @useData(data)
    def test_package_import(self, pkg, fail=False, alternative=None):
        run_python_package_import_test(self, pkg, "PYTHON3", fail, alternative)

if __name__ == '__main__':
    udf.main()
