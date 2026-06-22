#!/usr/bin/env python3

from exasol_python_test_framework import udf


class ExportAliasRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_expal CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_expal_data CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_expal")
        self.query("CREATE SCHEMA gr_expal_data")
        self.query("CREATE TABLE gr_expal_data.t(a INT, z VARCHAR(100))")
        self.query("INSERT INTO gr_expal_data.t VALUES (1, 'x')")

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

    def test_export_use_params(self):
        rows = self.query("""
            EXPORT gr_expal_data.t
            INTO SCRIPT gr_expal.expal_use_params
            WITH FOO='bar' BAR='foo'
        """)
        self.assertRowsEqual([(1,)], rows)


if __name__ == "__main__":
    udf.main()
