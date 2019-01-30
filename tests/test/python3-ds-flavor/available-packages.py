#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class AvailablePythonPackages(udf.TestCase):
    def setUp(self): 
        self.query('create schema available_packages', ignore_errors=True) 

    def import_test(self, pkg, fail=False, alternative=None):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT available_packages.test_import_of_package() returns int AS
            import %s
            def run(ctx): return 1
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
    def test_01(self): self.import_test('keras')
    def test_02(self): self.import_test('tensorflow')
    def test_03(self): self.import_test('kmodes')
    def test_04(self): self.import_test('seaborn')
    def test_05(self): self.import_test('matplotlib')
    def test_06(self): self.import_test('imblearn')
    def test_07(self): self.import_test('lifelines')
    def test_08(self): self.import_test('nltk')
    def test_09(self): self.import_test('gensim')
    def test_10(self): self.import_test('lxml')
    def test_11(self): self.import_test('ujson')
    def test_12(self): self.import_test('numpy')
    def test_13(self): self.import_test('scipy')
    def test_14(self): self.import_test('sklearn')
    def test_15(self): self.import_test('statsmodels')
    def test_16(self): self.import_test('joblib')
    def test_17(self): self.import_test('pandas')
    def test_18(self): self.import_test('pyexasol')
    def test_19(self): self.import_test('pycurl')


if __name__ == '__main__':
    udf.main()

