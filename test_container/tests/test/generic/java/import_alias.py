#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest


class _JavaUdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
class ImportAliasTest(_JavaUdfSetup):
    def setUp(self):
        super().setUp()
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create or replace table fn2.t(z varchar(3000))')
        self.query('create or replace table fn2.t2(y varchar(2000), z varchar(3000))')
        self.query('''
                   create connection FOOCONN to 'a' user 'b' identified by 'c'
                   ''', ignore_errors=True)

