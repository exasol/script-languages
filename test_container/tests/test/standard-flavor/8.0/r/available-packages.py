#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.available_python_packages_utils import run_python_package_import_test

AVAILABLE_PACKAGES_SCHEMA = "available_packages"


class AvailableRPackages(udf.TestCase):
    def setUp(self): 
        self.query(f'create schema {AVAILABLE_PACKAGES_SCHEMA}', ignore_errors=True)

    data = [
            ("acepack",),
            ("BradleyTerry2",),
            ("brglm",),
            ("caret",),
            ("chron",),
            ("data.table",),
            ("digest",),
            ("e1071",),
            ("fastcluster",),
            ("flashClust",),
            ("foreach",),
            ("Formula",),
            ("gbm",),
            ("gtools",),
            ("htmltools",),
            ("iterators",),
            ("lme4",),
            ("magrittr",),
            ("minqa",),
            ("nloptr",),
            ("plyr",),
            ("profileModel",),
            ("proto",),
            ("randomForest",),
            ("Rcpp",),
            ("RCurl",),
            ("reshape2",),
            ("RODBC",),
            ("redux",),
            ("scales",),
            ("stringr",),
            ("XML",),
            ("dplyr",),
            ("jsonlite",),
            ("purrr",),
            ("rjson",),
            ("tidyr",),
            ("tibble",),
            ("yaml",),
            ("httr",),
            ("glue",),
            ("oysteR",),
            ("SparseM",),
            ("caretEnsemble",),
        ]

    @useData(data)
    def test_package_import(self, pkg, fail=False, alternative=None):
        self.query(udf.fixindent(f'''
            CREATE OR REPLACE R SCALAR SCRIPT {AVAILABLE_PACKAGES_SCHEMA}.test_import_of_package() returns int AS
            library(%s)
            run <- function(ctx) {{ return(1) }}
            /
            ''' % (pkg)))
        try:
            rows = self.query(f'''SELECT {AVAILABLE_PACKAGES_SCHEMA}.test_import_of_package() FROM dual''')
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

