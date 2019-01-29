#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class AvailablePythonPackages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True) 

    def import_test(self, pkg, alternative=None):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT available_packages.test_import_of_package() returns int AS
            import %s
            def run(ctx): return 1
            /
            ''' % (pkg)))
        try:
            rows = self.query('''SELECT available_packages.test_import_of_package() FROM dual''')
            self.assertRowsEqual([(1,)], rows)
        except:
            if alternative:
                self.import_test(alternative)
            else:
                raise

               

    def test_01(self): self.import_test('cffi')
    def test_02(self): self.import_test('cryptography')
    def test_03(self): self.import_test('docutils')
    def test_04(self): self.import_test('enum')
    def test_05(self): self.import_test('idna')
    def test_06(self): self.import_test('ipaddress')
    def test_07(self): self.import_test('jinja2')
    def test_08(self): self.import_test('martian')
    def test_09(self): self.import_test('google.protobuf')
    def test_10(self): self.import_test('pyasn1')
    def test_11(self): self.import_test('pycparser')
    def test_12(self): self.import_test('pycryptopp')
    def test_13(self): self.import_test('pyftpdlib')
    def test_14(self): self.import_test('pygments')
    def test_15(self): self.import_test('pykickstart')
    def test_16(self): self.import_test('pyodbc')
    def test_17(self): self.import_test('OpenSSL')
    def test_18(self): self.import_test('pyPdf','PyPDF2')
    def test_19(self): self.import_test('ldb')
    def test_20(self): self.import_test('ldap')
    def test_21(self): self.import_test('roman')
    def test_22(self): self.import_test('samba')
    def test_23(self): self.import_test('sklearn')
    def test_24(self): self.import_test('talloc')
    def test_25(self): self.import_test('cjson')
    def test_26(self): self.import_test('lxml')
    def test_27(self): self.import_test('numpy')
    def test_28(self): self.import_test('setuptools')
    def test_29(self): self.import_test('pandas')
    def test_30(self): self.import_test('redis')
    def test_31(self): self.import_test('scipy')
    def test_32(self): self.import_test('boto')
    def test_33(self): self.import_test('pycurl')
    def test_34(self): self.import_test('requests')


if __name__ == '__main__':
    udf.main()

