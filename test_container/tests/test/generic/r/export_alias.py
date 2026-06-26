#!/usr/bin/env python3

from exasol_python_test_framework import udf


class ExportAliasRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_expal CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_expal_data CASCADE", ignore_errors=True)
        self.query("DROP CONNECTION gr_expal_fooconn", ignore_errors=True)
        self.query("CREATE SCHEMA gr_expal")
        self.query("CREATE SCHEMA gr_expal_data")
        self.query("CREATE TABLE gr_expal_data.t(a INT, z VARCHAR(100))")
        self.query("INSERT INTO gr_expal_data.t VALUES (1, 'x')")
        self.query("CREATE TABLE gr_expal_data.tl(a INT, z VARCHAR(100))")
        self.query("INSERT INTO gr_expal_data.tl VALUES (1, 'x')")
        self.query("CREATE CONNECTION gr_expal_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_test_pass_fail(res VARCHAR(100))
            EMITS (x INT) AS
            run <- function(ctx) {
                if (ctx$res == 'ok') {
                    ctx$emit(1L)
                } else {
                    ctx$emit(2L)
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_use_params(...)
            RETURNS INT AS
            generate_sql_for_export_spec <- function(export_spec) {
                if (export_spec$parameters[['FOO']] == 'bar' && export_spec$parameters[['BAR']] == 'foo') {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('ok')")
                } else {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('failed')")
                }
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_use_connection_name(...)
            RETURNS INT AS
            generate_sql_for_export_spec <- function(export_spec) {
                if (!is.null(export_spec$connection_name) &&
                    export_spec$connection_name == 'GR_EXPAL_FOOCONN' &&
                    export_spec$parameters[['FOO']] == 'bar') {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('ok')")
                } else {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('failed')")
                }
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_use_connection_info(...)
            RETURNS INT AS
            generate_sql_for_export_spec <- function(export_spec) {
                conn <- export_spec$connection
                if (!is.null(conn) &&
                    conn$type == 'password' &&
                    conn$address == 'a' &&
                    conn$user == 'b' &&
                    conn$password == 'c') {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('ok')")
                } else {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('failed')")
                }
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_use_has_truncate(...)
            RETURNS INT AS
            generate_sql_for_export_spec <- function(export_spec) {
                if (isTRUE(export_spec$has_truncate)) {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('ok')")
                } else {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('failed')")
                }
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_use_replace_created_by(...)
            RETURNS INT AS
            generate_sql_for_export_spec <- function(export_spec) {
                if (isTRUE(export_spec$has_replace) &&
                    !is.null(export_spec$created_by)) {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('ok')")
                } else {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('failed')")
                }
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_use_column_names(...)
            RETURNS INT AS
            generate_sql_for_export_spec <- function(export_spec) {
                if (length(export_spec$source_column_names) >= 2) {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('ok')")
                } else {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('failed')")
                }
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_use_column_name_lower_case(...)
            RETURNS INT AS
            generate_sql_for_export_spec <- function(export_spec) {
                n <- export_spec$source_column_names
                if (length(n) >= 2 && n[[2]] == 'z') {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('ok')")
                } else {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('failed')")
                }
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_expal.expal_use_column_selection(...)
            RETURNS INT AS
            generate_sql_for_export_spec <- function(export_spec) {
                n <- export_spec$source_column_names
                if (length(n) == 2 && toupper(n[[1]]) == 'A' && n[[2]] == 'z') {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('ok')")
                } else {
                    paste0("select ", exa$meta$script_schema, ".expal_test_pass_fail('failed')")
                }
            }
            /
        """))

    def tearDown(self):
        self.query("DROP CONNECTION gr_expal_fooconn", ignore_errors=True)

    def test_export_use_params(self):
        rows = self.query("""
            EXPORT gr_expal_data.t
            INTO SCRIPT gr_expal.expal_use_params
            WITH FOO='bar' BAR='foo'
        """)
        self.assertRowsEqual([(1,)], rows)

    def test_export_use_connection_name(self):
        rows = self.query("""
            EXPORT gr_expal_data.t
            INTO SCRIPT gr_expal.expal_use_connection_name AT GR_EXPAL_FOOCONN
            WITH FOO='bar' BAR='foo'
        """)
        self.assertRowsEqual([(1,)], rows)

    def test_export_use_connection_info(self):
        rows = self.query("""
            EXPORT gr_expal_data.t
            INTO SCRIPT gr_expal.expal_use_connection_info AT 'a' USER 'b' IDENTIFIED BY 'c'
            WITH FOO='bar' BAR='foo'
        """)
        self.assertRowsEqual([(1,)], rows)

    def test_export_use_has_truncate(self):
        rows = self.query("""
            EXPORT gr_expal_data.t
            INTO SCRIPT gr_expal.expal_use_has_truncate
            WITH FOO='bar' BAR='foo' TRUNCATE
        """)
        self.assertRowsEqual([(1,)], rows)

    def test_export_use_replace_created_by(self):
        rows = self.query("""
            EXPORT gr_expal_data.t
            INTO SCRIPT gr_expal.expal_use_replace_created_by
            WITH FOO='bar' BAR='foo' REPLACE CREATED BY 'create table t(a int, z varchar(100))'
        """)
        self.assertRowsEqual([(1,)], rows)

    def test_export_use_column_names(self):
        rows = self.query("""
            EXPORT gr_expal_data.t
            INTO SCRIPT gr_expal.expal_use_column_names
            WITH FOO='bar' BAR='foo'
        """)
        self.assertRowsEqual([(1,)], rows)

    def test_export_use_query(self):
        rows = self.query("""
            EXPORT (SELECT a AS col1, z AS col2 FROM gr_expal_data.t)
            INTO SCRIPT gr_expal.expal_use_column_names
            WITH FOO='bar' BAR='foo'
        """)
        self.assertRowsEqual([(1,)], rows)

    def test_export_use_column_name_lower_case(self):
        rows = self.query("""
            EXPORT gr_expal_data."tl"
            INTO SCRIPT gr_expal.expal_use_column_name_lower_case
            WITH FOO='bar' BAR='foo'
        """)
        self.assertRowsEqual([(1,)], rows)

    def test_export_use_column_selection(self):
        rows = self.query("""
            EXPORT gr_expal_data."tl"(a, "z")
            INTO SCRIPT gr_expal.expal_use_column_selection
            WITH FOO='bar' BAR='foo'
        """)
        self.assertRowsEqual([(1,)], rows)


if __name__ == "__main__":
    udf.main()
