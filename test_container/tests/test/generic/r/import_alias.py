#!/usr/bin/env python3

from exasol_python_test_framework import udf


class ImportAliasRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_impal CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_impal_data CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_impal")
        self.query("CREATE SCHEMA gr_impal_data")
        self.query("CREATE TABLE gr_impal_data.t2(y VARCHAR(2000), z VARCHAR(3000))")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_impal.impal_use_param_foo_bar(...)
            RETURNS VARCHAR(2000) AS
            generate_sql_for_import_spec <- function(import_spec) {
                paste0(
                    "select '",
                    import_spec$parameters[['FOO']],
                    "', '",
                    import_spec$parameters[['BAR']],
                    "'"
                )
            }
            /
        """))

    def test_import_use_params(self):
        self.query("""
            IMPORT INTO gr_impal_data.t2
            FROM SCRIPT gr_impal.impal_use_param_foo_bar
            WITH FOO='bar' BAR='foo'
        """)
        rows = self.query("SELECT * FROM gr_impal_data.t2")
        self.assertRowsEqual([('bar', 'foo')], rows)

    def test_import_use_params_subselect(self):
        rows = self.query("""
            SELECT * FROM (
                IMPORT FROM SCRIPT gr_impal.impal_use_param_foo_bar
                WITH FOO='bar' BAR='foo'
            )
        """)
        self.assertRowsEqual([('bar', 'foo')], rows)


if __name__ == "__main__":
    udf.main()
