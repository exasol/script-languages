#!/usr/bin/env python3


from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.available_python_packages_utils import run_python_package_import_test

class AvailablePythonPackages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True)

    data = [
            ("cffi",),
            ("cryptography",),
            ("docutils",),
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
            ("ldb",),
            ("roman",),
            ("sklearn",),
            ("cjson",),
            ("lxml",),
            ("numpy",),
            ("setuptools",),
            ("pandas",),
            ("redis",),
            ("scipy",),
            ("boto3",),
            ("boto",),
            ("pycurl",),
            ("requests",),
            ("EXASOL",),
            ("paramiko",),
            ("pysftp",),
            ("samba",),
            ("simplejson",),
            ("pycparser",),
            ("pycryptopp",),
            ("pygments",),
            ("pykickstart",),
            ("talloc",),
            ("pyPdf",False,"PyPDF2"),
        ]

    @useData(data)
    def test_package_import(self, pkg, fail=False, alternative=None):
        run_python_package_import_test(self, pkg, "PYTHON", fail, alternative)

class AvailablePython3Packages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True)

    data = [
            ("cffi",),
            ("cryptography",),
            ("docutils",),
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
            ("boto",),
            ("pycurl",),
            ("requests",),
            ("EXASOL",),
            ("pyexasol",),
            ("paramiko",),
            ("pysftp",),
            ("samba",),
            ("simplejson",),
            ("pycparser",),
            ("pygments",),
            ("pykickstart",),
            ("simplejson",),
            ("pyPdf",False,"PyPDF2"),
        ]

    @useData(data)
    def test_package_import(self, pkg, fail=False, alternative=None):
        run_python_package_import_test(self, pkg, "PYTHON3", fail, alternative)

class AvailableRPackages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True) 

    def import_test(self, pkg, fail=False, alternative=None):
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

    def test_00(self): self.import_test('unknown_package',True)
    def test_01(self): self.import_test('RCurl')
    def test_02(self): self.import_test('XML')
    def test_03(self): self.import_test('Formula')
    def test_04(self): self.import_test('RColorBrewer')
    def test_05(self): self.import_test('RODBC')
    def test_06(self): self.import_test('acepack')
    def test_07(self): self.import_test('chron')
    def test_08(self): self.import_test('e1071')
    def test_09(self): self.import_test('fastcluster')
    def test_10(self): self.import_test('gbm')
    def test_11(self): self.import_test('gridExtra')
    def test_12(self): self.import_test('gtable')
    def test_13(self): self.import_test('latticeExtra')
    def test_14(self): self.import_test('randomForest')
    def test_15(self): self.import_test('BradleyTerry2')
    def test_16(self): self.import_test('brglm')
    def test_17(self): self.import_test('car')
    def test_18(self): self.import_test('caret')
    def test_19(self): self.import_test('colorspace')
    def test_20(self): self.import_test('dichromat')
    def test_21(self): self.import_test('digest')
    def test_22(self): self.import_test('foreach')
    def test_23(self): self.import_test('ggplot2')
    def test_24(self): self.import_test('gtools')
    def test_25(self): self.import_test('iterators')
    def test_26(self): self.import_test('labeling')
    def test_27(self): self.import_test('lme4')
    def test_28(self): self.import_test('magrittr')
    def test_29(self): self.import_test('minqa')
    def test_30(self): self.import_test('munsell')
    def test_31(self): self.import_test('nloptr')
    def test_32(self): self.import_test('plyr')
    def test_33(self): self.import_test('profileModel')
    def test_34(self): self.import_test('Rcpp')
    def test_35(self): self.import_test('RcppEigen')
    def test_36(self): self.import_test('reshape2')
    def test_37(self): self.import_test('scales')
    def test_38(self): self.import_test('stringr')
    def test_39(self): self.import_test('rredis')
    def test_40(self): self.import_test('data.table')
    def test_41(self): self.import_test('htmltools')
    def test_42(self): self.import_test('flashClust')


if __name__ == '__main__':
    udf.main()

