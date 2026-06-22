#!/usr/bin/env python3

from exasol_python_test_framework import udf


class ImportAliasRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_impal CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_impal_data CASCADE", ignore_errors=True)
        self.query("DROP CONNECTION gr_impal_fooconn", ignore_errors=True)
        self.query("CREATE SCHEMA gr_impal")
        self.query("CREATE SCHEMA gr_impal_data")
        self.query("CREATE TABLE gr_impal_data.t(z VARCHAR(2000))")
        self.query("CREATE TABLE gr_impal_data.t2(y VARCHAR(2000), z VARCHAR(3000))")
        self.query("CREATE CONNECTION gr_impal_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'")

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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_impal.impal_use_is_subselect(...)
            EMITS (x VARCHAR(2000)) AS
            generate_sql_for_import_spec <- function(import_spec) {
                paste0("select '", tolower(as.character(import_spec$is_subselect)), "'")
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_impal.impal_use_connection_name(...)
            EMITS (x VARCHAR(2000)) AS
            generate_sql_for_import_spec <- function(import_spec) {
                paste0("select '", import_spec$connection_name, "'")
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_impal.impal_use_connection_fooconn(...)
            EMITS (x VARCHAR(2000)) AS
            generate_sql_for_import_spec <- function(import_spec) {
                c <- exa$get_connection('GR_IMPAL_FOOCONN')
                paste0("select '", c$address, c$user, c$password, "'")
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_impal.impal_use_connection(...)
            EMITS (x VARCHAR(2000)) AS
            generate_sql_for_import_spec <- function(import_spec) {
                conn <- import_spec$connection
                paste0("select '", conn$user, conn$password, conn$address, conn$type, "'")
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_impal.impal_use_all(...)
            EMITS (x VARCHAR(2000)) AS
            generate_sql_for_import_spec <- function(import_spec) {
                is_sub <- toupper(as.character(import_spec$is_subselect))
                conn_str <- 'X'
                conn_name <- 'Y'
                foo <- 'Z'
                if (!is.null(import_spec$connection)) {
                    c <- import_spec$connection
                    conn_str <- paste0(c$user, c$password, c$address, c$type)
                }
                if (!is.null(import_spec$connection_name)) {
                    conn_name <- import_spec$connection_name
                }
                if (!is.null(import_spec$parameters[['FOO']])) {
                    foo <- import_spec$parameters[['FOO']]
                }
                paste0("select 1, '", is_sub, '_', conn_name, '_', conn_str, '_', foo, "'")
            }
            /
        """))

    def tearDown(self):
        self.query("DROP CONNECTION gr_impal_fooconn", ignore_errors=True)

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

    def test_import_use_is_subselect(self):
        self.query("""
            IMPORT INTO gr_impal_data.t FROM SCRIPT gr_impal.impal_use_is_subselect
        """)
        rows = self.query("SELECT * FROM gr_impal_data.t")
        self.assertRowsEqual([('false',)], rows)
        self.query("TRUNCATE TABLE gr_impal_data.t")

    def test_import_use_is_subselect_subselect(self):
        rows = self.query("""
            SELECT * FROM (IMPORT FROM SCRIPT gr_impal.impal_use_is_subselect)
        """)
        self.assertRowsEqual([('true',)], rows)

    def test_import_use_connection_name(self):
        self.query("IMPORT INTO gr_impal_data.t FROM SCRIPT gr_impal.impal_use_connection_name AT GR_IMPAL_FOOCONN")
        rows = self.query("SELECT * FROM gr_impal_data.t")
        self.assertRowsEqual([('GR_IMPAL_FOOCONN',)], rows)
        self.query("TRUNCATE TABLE gr_impal_data.t")

    def test_import_use_connection_name_subselect(self):
        rows = self.query("""
            SELECT * FROM (IMPORT FROM SCRIPT gr_impal.impal_use_connection_name AT GR_IMPAL_FOOCONN)
        """)
        self.assertRowsEqual([('GR_IMPAL_FOOCONN',)], rows)

    def test_import_use_connection_fooconn(self):
        rows = self.query("IMPORT FROM SCRIPT gr_impal.impal_use_connection_fooconn")
        self.assertRowsEqual([('abc',)], rows)

    def test_import_use_connection(self):
        self.query("""
            IMPORT INTO gr_impal_data.t FROM SCRIPT gr_impal.impal_use_connection
            AT 'a' USER 'hans' IDENTIFIED BY 'meiser'
        """)
        rows = self.query("SELECT * FROM gr_impal_data.t")
        self.assertRowsEqual([('hansmeiserapassword',)], rows)
        self.query("TRUNCATE TABLE gr_impal_data.t")

    def test_import_use_connection_subselect(self):
        rows = self.query("""
            SELECT * FROM (
                IMPORT FROM SCRIPT gr_impal.impal_use_connection
                AT 'a' USER 'hans' IDENTIFIED BY 'meiser'
            )
        """)
        self.assertRowsEqual([('hansmeiserapassword',)], rows)

    def test_import_use_all(self):
        self.query("""
            IMPORT INTO gr_impal_data.t2 FROM SCRIPT gr_impal.impal_use_all
            AT 'a' USER 'hans' IDENTIFIED BY 'meiser' WITH FOO='a value'
        """)
        rows = self.query("SELECT * FROM gr_impal_data.t2")
        self.assertTrue(len(rows) == 1)
        self.assertIn('a value', rows[0][1])
        self.query("TRUNCATE TABLE gr_impal_data.t2")


if __name__ == "__main__":
    udf.main()
