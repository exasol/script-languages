#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.available_python_packages_utils import run_python_package_import_test

AVAILABLE_PACKAGES_SCHEMA = "available_packages"


class AvailablePython3Packages(udf.TestCase):
    def setUp(self): 
        self.query(f'create schema {AVAILABLE_PACKAGES_SCHEMA}', ignore_errors=True)

    data = [
            ("google.cloud.asset",),
            ("google.cloud.bigquery",),
            ("google.cloud.bigquery_storage",),
            ("google.cloud.bigtable",),
            ("google.cloud.devtools.containeranalysis_v1",),
            ("google.cloud.datacatalog",),
            ("google.cloud.datastore",),
            ("google.cloud.firestore",),
            ("google.cloud.kms",),
            ("google.cloud.logging",),
            ("google.cloud.monitoring",),
#            ("google.cloud.ndb",), # fails at input
            ("google.cloud.pubsub",),
            ("google.cloud.spanner",),
            ("google.cloud.storage",),
            ("google.cloud.trace",),
        ]

    @useData(data)
    def test_package_import(self, pkg, fail=False, alternative=None):
        run_python_package_import_test(self, AVAILABLE_PACKAGES_SCHEMA, "PYTHON3", pkg, fail, alternative)


if __name__ == '__main__':
    udf.main()
