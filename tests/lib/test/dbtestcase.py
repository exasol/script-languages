#!/usr/bin/env python2.7

import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(__file__, '../../../lib')))

import exatest
from exatest.utils import chdir
from exatest import (
        useData,
        )

class TestParameterized(exatest.TestCase):

    @useData((x,) for x in range(10))
    def test_parameterized(self, x):
        self.assertRowsEqual([(None,)], self.query('select * from dual'))

    @useData((x,) for x in range(1000))
    def test_large_parameterized(self, x):
        self.assertRowsEqual([(None,)], self.query('select * from dual'))

class TestSetUp(exatest.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA t1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA t1')

    def test_1(self):
        self.query('select * from dual')

    def test_2(self):
        self.query('select * from dual')

class ODBCTest(exatest.TestCase):
    def test_find_odbcini_after_chdir(self):
        self.assertTrue(os.path.exists('odbc.ini'))
        with chdir('/'):
            self.assertFalse(os.path.exists('odbc.ini'))
            self.query('select * from dual')

if __name__ == '__main__':
    # remove undefined option used in wrapper script
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith('--jdbc-path='):
            # --foo=bar
            sys.argv.pop(i)
            break
        if sys.argv[i].startswith('--jdbc-path'):
            # --foo bar
            sys.argv.pop(i)
            sys.argv.pop(i)
            break

    exatest.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
