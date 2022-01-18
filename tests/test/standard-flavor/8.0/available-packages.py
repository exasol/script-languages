#!/usr/bin/env python3


from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData

class AvailablePython3Packages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True)

    data = [
            ("cffi",),
            ("cryptography",),
            ("enum",),
            ("idna",),
            ("ipaddress",),
            ("jinja2",),
            ("martian",),
            ("google.protobuf",),
            ("pyasn1",),
            ("pyftpdlib",),
            ("pyodbc",),
            ("OpenSSL",),
            ("ldap",),
            ("roman",),
            ("sklearn",),
            ("ujson",),
            ("lxml",),
            ("numpy",),
            ("setuptools",),
            ("pandas",),
            ("redis",),
            ("scipy",),
            ("boto3",),
            ("pycurl",),
            ("requests",),
            ("pyexasol",),
            ("EXASOL",),
            ("paramiko",),
            ("pysftp",),
            ("simplejson",),
            ("simdjson",),
            ("pybase64",),
            ("xgboost",),
            ("exasol_bucketfs_utils_python",),
            ("yaml",),
            ("bitsets",),
            ("pybloomfilter",),
            ("bitarray",),
            ("pyarrow",),
        ]

    @useData(data)
    def test_package_import(self, pkg, fail=False, alternative=None):
        sql=udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT available_packages.test_import_of_package() returns int AS
            
            import %s
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

class AvailableRPackages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True)

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
            ("RcppEigen",),
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
        self.query(udf.fixindent('''
            CREATE OR REPLACE R SCALAR SCRIPT available_packages.test_import_of_package() returns int AS
            library(%s)
            run <- function(ctx) { return(1) }
            /
            ''' % (pkg)))
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

