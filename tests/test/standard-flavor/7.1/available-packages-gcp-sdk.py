#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData


class AvailablePythonPackages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True) 

    data = [
            ("from google.cloud import asset",),
            ("from google.cloud import bigquery",),
            ("from google.cloud import bigquery_storage",),
            ("from google.cloud import bigtable",),
            ("from google.cloud.devtools import containeranalysis_v1",),
            ("from google.cloud import datacatalog",),
            ("from google.cloud import datastore",),
            ("from google.cloud import firestore",),
            ("from google.cloud import kms",),
            ("from google.cloud import logging",),
            ("from google.cloud import monitoring",),
#            ("from google.cloud import ndb",), # fails at input
            ("from google.cloud import pubsub",),
            ("from google.cloud import spanner",),
            ("from google.cloud import storage",),
            ("from google.cloud import trace",),
        ]

    @useData(data)
    def test_package_import(self, pkg, fail=False, alternative=None):
        sql=udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT available_packages.test_import_of_package() returns int AS
            
            %s
            def run(ctx): return 1
            /
            ''' % (pkg))
        print(sql)
        self.query(sql)
        try:
            rows = self.query('''SELECT available_packages.test_import_of_package() FROM dual''')
            if not fail:
                self.assertRowsEqual([(1,)], rows)
            else:
                assert 'Expected Failure' == 'not found'
        except:
            if fail:
                return
            if alternative:
                self.import_test(alternative,fail)
            else:
                raise


if __name__ == '__main__':
    udf.main()
