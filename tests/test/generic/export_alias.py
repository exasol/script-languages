#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import requires
import exatest

# ATTENTION!
# The logic for the tests had to be put in the export_alias.sql files for each language.
# This was required because EXPORT INTO SCRIPT can only return a single integer.

class ExportAliasTest(udf.TestCase):
    result_unknown = 0
    result_ok = 1
    result_failed = 2
    result_test_error = 3

    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create or replace table t(a int, z varchar(3000))')
        self.query("insert into t values (1, 'x')")
        self.query('create or replace table "tl"(a int, "z" varchar(3000))')
        self.query("insert into \"tl\" values (1, 'x')")
        self.query("create connection FOOCONN to 'a' user 'b' identified by 'c'", ignore_errors=True)

    @requires('EXPAL_TEST_PASS_FAIL')
    @requires('EXPAL_USE_PARAM_FOO_BAR')
    def test_export_use_params(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_param_foo_bar with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    @requires('EXPAL_TEST_PASS_FAIL')
    @requires('EXPAL_USE_CONNECTION_NAME')
    def test_export_use_connection_name(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_connection_name AT FOOCONN with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    @requires('EXPAL_TEST_PASS_FAIL')
    @requires('EXPAL_USE_CONNECTION_INFO')
    def test_export_use_connection_info(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_connection_info AT 'a' USER 'b' IDENTIFIED BY 'c' with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    @requires('EXPAL_TEST_PASS_FAIL')
    @requires('EXPAL_USE_HAS_TRUNCATE')
    def test_export_use_has_truncate(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_has_truncate with foo='bar' bar='foo' truncate")
        self.assertEqual(self.result_ok, rows)

    @requires('EXPAL_TEST_PASS_FAIL')
    @requires('EXPAL_USE_REPLACE_CREATED_BY')
    def test_export_use_replace_created_by(self):
        rows = self.executeStatement("EXPORT fn2.t INTO SCRIPT fn1.expal_use_replace_created_by with foo='bar' bar='foo' replace created by 'create table t(a int, z varchar(3000))'")
        self.assertEqual(self.result_ok, rows)

    @requires('EXPAL_TEST_PASS_FAIL')
    @requires('EXPAL_USE_COLUMN_NAME_LOWER_CASE')
    def test_export_use_column_name_lower_case(self):
        rows = self.executeStatement("EXPORT fn2.\"tl\" INTO SCRIPT fn1.expal_use_column_name_lower_case with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    @requires('EXPAL_TEST_PASS_FAIL')
    @requires('EXPAL_USE_COLUMN_SELECTION')
    def test_export_use_column_selection(self):
        rows = self.executeStatement("EXPORT fn2.\"tl\"(a, \"z\") INTO SCRIPT fn1.expal_use_column_selection with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

    @requires('EXPAL_TEST_PASS_FAIL')
    @requires('EXPAL_USE_QUERY')
    def test_export_use_query(self):
        rows = self.executeStatement("EXPORT (select a as 'col1', \"z\" as 'col2' from fn2.\"tl\") INTO SCRIPT fn1.expal_use_query with foo='bar' bar='foo'")
        self.assertEqual(self.result_ok, rows)

if __name__ == '__main__':
    udf.main()
