#!/usr/bin/env python3

from exasol_python_test_framework import udf


class _JavaUdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
# ATTENTION!
# The logic for the tests had to be put in the export_alias.sql files for each language.
# This was required because EXPORT INTO SCRIPT can only return a single integer.


class ExportAliasTest(_JavaUdfSetup):
    result_unknown = 0
    result_ok = 1
    result_failed = 2
    result_test_error = 3

    def setUp(self):
        super().setUp()
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create or replace table fn2.t(a int, z varchar(3000))')
        self.query("insert into fn2.t values (1, 'x')")
        self.query('create or replace table fn2.\"tl\"(a int, \"z\" varchar(3000))')
        self.query("insert into fn2.\"tl\" values (1, 'x')")
        self.query("create connection FOOCONN to 'a' user 'b' identified by 'c'", ignore_errors=True)
