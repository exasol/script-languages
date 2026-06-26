#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest
from exasol_python_test_framework.udf import skip


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
                types <- 'T'
                names <- 'N'
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
                if (!is.null(import_spec$subselect_column_types)) {
                    for (i in seq_along(import_spec$subselect_column_types)) {
                        types <- paste0(types, import_spec$subselect_column_types[[i]])
                        names <- paste0(names, import_spec$subselect_column_names[[i]])
                    }
                }
                paste0("select 1, '", is_sub, '_', conn_name, '_', conn_str, '_', foo, '_', types, '_', names, "'")
            }
            /
        """))

    def tearDown(self):
        self.query("DROP CONNECTION gr_impal_fooconn", ignore_errors=True)

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid=username, pwd=password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username=username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username=username, password=password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

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
        rows = self.query("""
            IMPORT FROM SCRIPT gr_impal.impal_use_param_foo_bar
            WITH FOO='bar' BAR='foo'
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
        rows = self.query("""
            IMPORT FROM SCRIPT gr_impal.impal_use_is_subselect
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
        rows = self.query("""
            IMPORT FROM SCRIPT gr_impal.impal_use_connection_name AT GR_IMPAL_FOOCONN
        """)
        self.assertRowsEqual([('GR_IMPAL_FOOCONN',)], rows)

    def test_import_use_connection_fooconn(self):
        rows = self.query("IMPORT FROM SCRIPT gr_impal.impal_use_connection_fooconn")
        self.assertRowsEqual([('abc',)], rows)

    def test_import_use_connection_fooconn_fails_for_user_foo(self):
        self.createUser('foo', 'foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        with self.assertRaisesRegex(Exception, 'insufficient privileges'):
            foo_conn.query('IMPORT FROM SCRIPT gr_impal.impal_use_connection_fooconn')
        self.query('DROP USER foo CASCADE')

    @skip("IMPORT FROM SCRIPT cannot be used in view definitions")
    def test_import_use_connection_fooconn_for_user_foo_and_view(self):
        self.query('CREATE VIEW gr_impal_data.fooconn_import_view AS IMPORT FROM SCRIPT gr_impal.impal_use_connection_fooconn')
        self.createUser('foo', 'foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        rows = foo_conn.query('SELECT * FROM gr_impal_data.fooconn_import_view')
        self.assertRowsEqual([('abc',)], rows)
        self.query('DROP USER foo CASCADE')
        self.query('DROP VIEW gr_impal_data.fooconn_import_view')

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
        rows = self.query("""
            IMPORT FROM SCRIPT gr_impal.impal_use_connection
            AT 'a' USER 'hans' IDENTIFIED BY 'meiser'
        """)
        self.assertRowsEqual([('hansmeiserapassword',)], rows)

    def test_import_use_all(self):
        self.query("""
            IMPORT INTO gr_impal_data.t2 FROM SCRIPT gr_impal.impal_use_all
            AT 'a' USER 'hans' IDENTIFIED BY 'meiser' WITH FOO='a value'
        """)
        rows = self.query("SELECT * FROM gr_impal_data.t2")
        self.assertRowsEqual([('1', 'FALSE_Y_hansmeiserapassword_a value_T_N')], rows)
        self.query("TRUNCATE TABLE gr_impal_data.t2")

    def test_import_use_all_subselect(self):
        rows = self.query("""
            SELECT * FROM (
                IMPORT INTO (a DOUBLE, b VARCHAR(3000)) FROM SCRIPT gr_impal.impal_use_all
                AT 'a' USER 'hans' IDENTIFIED BY 'meiser' WITH FOO='a value'
            )
        """)
        self.assertRowsEqual([(1, 'TRUE_Y_hansmeiserapassword_a value_TDOUBLEVARCHAR(3000) UTF8_NAB')], rows)
        rows = self.query("""
            IMPORT INTO (a DOUBLE, b VARCHAR(3000)) FROM SCRIPT gr_impal.impal_use_all
            AT 'a' USER 'hans' IDENTIFIED BY 'meiser' WITH FOO='a value'
        """)
        self.assertRowsEqual([(1, 'TRUE_Y_hansmeiserapassword_a value_TDOUBLEVARCHAR(3000) UTF8_NAB')], rows)

    def test_prepared_statement_params(self):
        with self.assertRaisesRegex(Exception, "syntax error, unexpected '\\?'"):
            self.query("""
                SELECT * FROM (
                    IMPORT INTO (a DOUBLE, b VARCHAR(3000)) FROM SCRIPT gr_impal.impal_use_all
                    AT 'a' USER 'hans' IDENTIFIED BY 'meiser' WITH FOO=?
                )
            """, 'bar')

    def test_prepared_statement_conn(self):
        with self.assertRaisesRegex(Exception, "syntax error, unexpected '\\?'"):
            self.query("""
                SELECT * FROM (
                    IMPORT INTO (a DOUBLE, b VARCHAR(3000)) FROM SCRIPT gr_impal.impal_use_all
                    AT ? USER ? IDENTIFIED BY ? WITH FOO='bar'
                )
            """, 'a', 'hans', 'meiser')

    def test_import_in_lua_scripting(self):
        self.query("""
            CREATE OR REPLACE SCRIPT gr_impal_data.s1() AS
                res = pquery [[ IMPORT INTO gr_impal_data.t FROM SCRIPT gr_impal.impal_use_is_subselect ]]
        """)
        self.query("EXECUTE SCRIPT gr_impal_data.s1()")
        rows = self.query("SELECT * FROM gr_impal_data.t")
        self.assertRowsEqual([('false',)], rows)
        self.query("TRUNCATE TABLE gr_impal_data.t")


if __name__ == "__main__":
    udf.main()
