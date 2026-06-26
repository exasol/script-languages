#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest


class DynamicOutputRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_dynout CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_dynout_data CASCADE", ignore_errors=True)
        self.query("DROP CONNECTION SPOT4245", ignore_errors=True)
        self.query("CREATE SCHEMA gr_dynout")
        self.query("CREATE SCHEMA gr_dynout_data")
        self.query("CREATE TABLE gr_dynout_data.small(x VARCHAR(2000), y DOUBLE)")
        self.query("INSERT INTO gr_dynout_data.small VALUES ('abc', 2.2)")
        self.query("CREATE CONNECTION SPOT4245 TO 'a' USER 'b' IDENTIFIED BY 'c'")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_simple_set(a DOUBLE)
            EMITS (...) AS
            run <- function(ctx) {
                ctx$emit(1)
            }

            default_output_columns <- function() {
                'x DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_emit_input(...)
            EMITS (...) AS
            run <- function(ctx) {
                out <- list()
                for (i in 1:exa$meta$input_column_count) {
                    out[[i]] <- ctx[[i]]()
                }
                do.call(ctx$emit, out)
            }

            default_output_columns <- function() {
                'a VARCHAR(2000), b DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_simple_set(a DOUBLE)
            EMITS (...) AS
            run <- function(ctx) {
                ctx$emit(1)
            }

            default_output_columns <- function() {
                'x DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_dynout.varemit_simple_scalar(a DOUBLE)
            EMITS (...) AS
            run <- function(ctx) {
                ctx$emit(1)
            }

            default_output_columns <- function() {
                'x DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_dynout.varemit_simple_all_dyn(...)
            EMITS (...) AS
            run <- function(ctx) {
                ctx$emit(1)
            }

            default_output_columns <- function() {
                'x DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_simple_syntax_var(...)
            EMITS (...) AS
            run <- function(ctx) {
                ctx$emit(1)
            }

            default_output_columns <- function() {
                'x DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_generic_emit(a VARCHAR(100))
            EMITS (...) AS
            run <- function(ctx) {
                args <- rep(list(ctx$a), exa$meta$output_column_count)
                do.call(ctx$emit, args)
            }

            default_output_columns <- function() {
                'a VARCHAR(100)'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_all_generic(...)
            EMITS (...) AS
            run <- function(ctx) {
                args <- rep(list(ctx[[1]]()), exa$meta$output_column_count)
                do.call(ctx$emit, args)
            }

            default_output_columns <- function() {
                'a VARCHAR(100)'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_non_var_emit(...) EMITS (a DOUBLE) AS
            run <- function(ctx) {
                ctx$emit(1)
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_simple_returns(a INT) RETURNS INT AS
            run <- function(ctx) {
                1L
            }

            default_output_columns <- function() {
                'x DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_dynout.default_varemit_empty_def(a DOUBLE)
            EMITS (...) AS
            run <- function(ctx) {
                ctx$emit(1.4)
            }

            default_output_columns <- function() {
                ''
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.copy_relation(...)
            EMITS (...) AS
            run <- function(ctx) {
                out <- list()
                for (i in 1:exa$meta$input_column_count) {
                    out[[i]] <- ctx[[i]]()
                }
                do.call(ctx$emit, out)
            }

            default_output_columns <- function() {
                'a DOUBLE, b DOUBLE, c DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_metadata_set_emit(a DOUBLE)
            EMITS (...) AS
            run <- function(ctx) {
                v <- as.double(ctx$a)
                ctx$emit(as.character(exa$meta$output_column_count), v)
                for (i in 1:exa$meta$output_column_count) {
                    col <- exa$meta$output_columns[[i]]
                    ctx$emit(as.character(col$name), v)
                    ctx$emit(as.character(col$type), v)
                    ctx$emit(as.character(col$sql_type), v)
                    ctx$emit(as.character(if (is.null(col$precision)) 0 else col$precision), v)
                    ctx$emit(as.character(if (is.null(col$scale)) 0 else col$scale), v)
                    ctx$emit(as.character(if (is.null(col$length)) 0 else col$length), v)
                }
            }

            default_output_columns <- function() {
                'a VARCHAR(123), b DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_emit_input_with_meta_check(...)
            EMITS (...) AS
            run <- function(ctx) {
                out <- list()
                for (i in 1:exa$meta$input_column_count) {
                    out[[i]] <- ctx[[i]]()
                }
                do.call(ctx$emit, out)
            }

            default_output_columns <- function() {
                'a INT, b DOUBLE, c VARCHAR(100)'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_dynout.output_columns_as_in_connection_spot4245(a DOUBLE)
            EMITS (...) AS
            run <- function(ctx) {
                c <- exa$get_connection('SPOT4245')
                ctx$emit(as.double(ctx$a), as.double(ctx$a), as.double(ctx$a), as.double(ctx$a))
            }

            default_output_columns <- function() {
                c <- exa$get_connection('SPOT4245')
                paste0(toupper(c$type), ' DOUBLE, ', toupper(c$address), ' DOUBLE, ', toupper(c$user), ' DOUBLE, ', toupper(c$password), ' DOUBLE')
            }
            /
        """))

        self.query("CREATE TABLE gr_dynout_data.target (a INT, b DOUBLE, c VARCHAR(100))")
        self.query("CREATE TABLE gr_dynout_data.groupt(id INT, n DOUBLE, v VARCHAR(999))")
        self.query("INSERT INTO gr_dynout_data.groupt VALUES (1, 1, 'aa'), (1, 2, 'ab'), (2, 2, 'ba')")

    def tearDown(self):
        self.query("DROP CONNECTION SPOT4245", ignore_errors=True)

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid=username, pwd=password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username=username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username=username, password=password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

    def checkColumnNamesOfQuery(self, query, expected_rows):
        self.query('DROP TABLE IF EXISTS gr_dynout_data.targetcreated')
        self.query('CREATE TABLE gr_dynout_data.targetcreated AS ' + str(query))
        rows = self.query('DESCRIBE gr_dynout_data.targetcreated')
        for i in range(len(expected_rows)):
            self.assertRowEqual(expected_rows[i][0:2], rows[i][0:2])

    def test_default_dynamic_output_columns(self):
        rows = self.query("""
            SELECT gr_dynout.varemit_simple_set(1.2)
            FROM DUAL
        """)
        self.assertRowsEqual([(1.0,)], rows)

    def test_copy_input_relation(self):
        rows = self.query("""
            SELECT gr_dynout.varemit_emit_input(x, y)
            FROM gr_dynout_data.small
        """)
        self.assertRowsEqual([('abc', 2.2)], rows)

    def test_copy_relation(self):
        rows = self.query("""
            SELECT gr_dynout.copy_relation(1, 2, 3)
            FROM DUAL
        """)
        self.assertRowEqual((1.0, 2.0, 3.0), rows[0])

    def test_create_script_set(self):
        rows = self.query("""
            SELECT COUNT(*) FROM EXA_ALL_SCRIPTS
            WHERE SCRIPT_NAME = 'VAREMIT_SIMPLE_SET'
            AND SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT%VAREMIT_SIMPLE_SET%EMITS (...)%'
        """)
        self.assertRowEqual((1,), rows[0])

    def test_create_script_scalar(self):
        rows = self.query("""
            SELECT COUNT(*) FROM EXA_ALL_SCRIPTS
            WHERE SCRIPT_NAME = 'VAREMIT_SIMPLE_SCALAR'
            AND SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT%VAREMIT_SIMPLE_SCALAR%EMITS (...)%'
        """)
        self.assertRowEqual((1,), rows[0])

    def test_create_script_all_dyn(self):
        rows = self.query("""
            SELECT COUNT(*) FROM EXA_ALL_SCRIPTS
            WHERE SCRIPT_NAME = 'VAREMIT_SIMPLE_ALL_DYN'
            AND SCRIPT_TEXT LIKE 'CREATE % SCALAR SCRIPT%VAREMIT_SIMPLE_ALL_DYN%(...)%EMITS (...)%'
        """)
        self.assertRowEqual((1,), rows[0])

    def test_create_script_syntax_var(self):
        rows = self.query("""
            SELECT COUNT(*) FROM EXA_ALL_SCRIPTS
            WHERE SCRIPT_NAME = 'VAREMIT_SIMPLE_SYNTAX_VAR'
            AND SCRIPT_TEXT LIKE 'CREATE % SET SCRIPT%VAREMIT_SIMPLE_SYNTAX_VAR%(...)%EMITS (...)%'
        """)
        self.assertRowEqual((1,), rows[0])

    def test_generic_emit(self):
        rows = self.query("""
            SELECT gr_dynout.varemit_generic_emit('SUPERDYNAMIC') EMITS (a VARCHAR(100))
        """)
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    def test_all_generic(self):
        rows = self.query("""
            SELECT gr_dynout.varemit_all_generic('SUPERDYNAMIC') EMITS (a VARCHAR(100))
        """)
        self.assertRowEqual(('SUPERDYNAMIC',), rows[0])

    def test_correctness_emits_subquery(self):
        rows = self.query("""
            SELECT "A" || 'x' || "B" FROM (
            SELECT gr_dynout.varemit_generic_emit('SUPERDYNAMIC') EMITS (a VARCHAR(100), b VARCHAR(100)))
        """)
        self.assertRowEqual(('SUPERDYNAMICxSUPERDYNAMIC',), rows[0])

    def test_correctness_emits_with_grouping(self):
        rows = self.query("""
            SELECT 'X' || COUNT(a) || 'X' FROM (
                SELECT gr_dynout.varemit_generic_emit('SUPERDYNAMIC') EMITS (a VARCHAR(100))
                FROM gr_dynout_data.groupt GROUP BY id
            ) WHERE a = 'SUPERDYNAMIC'
        """)
        self.assertRowEqual(('X2X',), rows[0])

    def test_correctness_nested(self):
        rows = self.query("""
            SELECT gr_dynout.varemit_generic_emit(c || 'D') EMITS (d VARCHAR(100)) FROM (
              SELECT gr_dynout.varemit_generic_emit(b || 'C') EMITS (c VARCHAR(100)) FROM (
                SELECT gr_dynout.varemit_generic_emit(a || 'B') EMITS (b VARCHAR(100)) FROM (
                  SELECT gr_dynout.varemit_generic_emit('A') EMITS (a VARCHAR(100))
                )
              )
            )
        """)
        self.assertRowEqual(('ABCD',), rows[0])

    def test_metadata_correctness(self):
        rows = self.query("""
            SELECT gr_dynout.varemit_metadata_set_emit(1) EMITS (a VARCHAR(123), b DOUBLE)
            FROM DUAL
        """)
        self.assertRowEqual(('2', 1.0), rows[0])
        self.assertRowEqual(('A', 1.0), rows[1])
        self.assertTrue(rows[2][0] in ['character'])
        self.assertRowEqual(('VARCHAR(123) UTF8', 1.0), rows[3])
        self.assertRowEqual(('123', 1.0), rows[6])
        self.assertRowEqual(('B', 1.0), rows[7])
        self.assertTrue(rows[8][0] in ['double'])
        self.assertRowEqual(('DOUBLE', 1.0), rows[9])

    def test_error_emit_missing(self):
        with self.assertRaisesRegex(Exception, 'The script has dynamic return arguments'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1)")

    def test_error_empty_emit(self):
        with self.assertRaisesRegex(Exception, 'Empty return argument definition is not allowed'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1) EMITS ()")

    def test_error_empty_emit_2(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1) EMITS (a)")

    def test_error_wrong_emit(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1) EMITS (int)")

    def test_error_redundant_name(self):
        with self.assertRaisesRegex(Exception, 'Return argument A is declared more than once'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1) EMITS (a INT, b INT, a INT)")

    def test_error_non_var_emit(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition'):
            self.query("SELECT gr_dynout.varemit_non_var_emit(1) EMITS (a DOUBLE)")

    def test_error_non_var_emit_2(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition'):
            self.query("SELECT gr_dynout.varemit_non_var_emit(1) EMITS ()")

    def test_error_returns_not_supported(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition'):
            self.query("SELECT gr_dynout.varemit_simple_returns(1) EMITS (a INT)")

    def test_error_built_in_set_not_supported(self):
        with self.assertRaisesRegex(Exception, 'emits specification is not allowed for built-in functions'):
            self.query("SELECT AVG(a) EMITS(a INT) FROM (VALUES 1, 2, 3) AS t(a)")

    def test_error_built_in_scalar_not_supported(self):
        with self.assertRaisesRegex(Exception, 'emits specification is not allowed for built-in functions'):
            self.query("SELECT -ABS(a) EMITS(a INT) FROM (VALUES 1, 2, 3) AS t(a)")

    def test_empty_string_error(self):
        with self.assertRaisesRegex(Exception, 'Empty default output columns'):
            self.query("SELECT gr_dynout.default_varemit_empty_def(42.42)")
        rows = self.query("SELECT gr_dynout.default_varemit_empty_def(42.42) EMITS (x DOUBLE)")
        self.assertRowEqual((1.4,), rows[0])

    def test_insert_basic(self):
        self.query("DELETE FROM gr_dynout_data.target")
        self.query("""
            INSERT INTO gr_dynout_data.target
            SELECT gr_dynout.varemit_emit_input(1, CAST(1.1 AS DOUBLE), 'a')
        """)
        rows = self.query("SELECT * FROM gr_dynout_data.target")
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
        self.query("DELETE FROM gr_dynout_data.target")

    def test_insert_metadata_correctness(self):
        self.query("DELETE FROM gr_dynout_data.target")
        self.query("""
            INSERT INTO gr_dynout_data.target
            SELECT gr_dynout.varemit_emit_input_with_meta_check(CAST(2 AS INT), CAST(2.2 AS DOUBLE), CAST('b' AS VARCHAR(100)))
        """)
        rows = self.query("SELECT * FROM gr_dynout_data.target")
        self.assertRowEqual((2, 2.2, 'b'), rows[0])
        self.query("DELETE FROM gr_dynout_data.target")

    def test_insert_target_columns_change_order(self):
        self.query("DELETE FROM gr_dynout_data.target")
        self.query("""
            INSERT INTO gr_dynout_data.target (c, b, a)
            SELECT gr_dynout.varemit_emit_input('c', CAST(3.3 AS DOUBLE), 3)
        """)
        rows = self.query("SELECT * FROM gr_dynout_data.target")
        self.assertRowEqual((3, 3.3, 'c'), rows[0])
        self.query("DELETE FROM gr_dynout_data.target")

    def test_insert_target_columns_subset(self):
        self.query("DELETE FROM gr_dynout_data.target")
        self.query("""
            INSERT INTO gr_dynout_data.target (b)
            SELECT gr_dynout.varemit_emit_input(CAST(4.4 AS DOUBLE))
        """)
        rows = self.query("SELECT * FROM gr_dynout_data.target")
        self.assertRowEqual((None, 4.4, None), rows[0])
        self.query("DELETE FROM gr_dynout_data.target")

    def test_insert_emits_not_allowed(self):
        with self.assertRaisesRegex(Exception, 'The return arguments for EMITS functions are inferred'):
            self.query("INSERT INTO gr_dynout_data.target SELECT gr_dynout.varemit_emit_input(1) EMITS (a INT)")

    def test_dynamic_out_from_connection_SPOT4245(self):
        expected_rows = [
            ('PASSWORD', 'DOUBLE', 'TRUE', 'FALSE'),
            ('A', 'DOUBLE', 'TRUE', 'FALSE'),
            ('B', 'DOUBLE', 'TRUE', 'FALSE'),
            ('C', 'DOUBLE', 'TRUE', 'FALSE'),
        ]
        self.checkColumnNamesOfQuery(
            "SELECT gr_dynout.output_columns_as_in_connection_spot4245(1.0)",
            expected_rows,
        )

    def test_dynamic_out_from_connection_SPOT4245_fails_for_user_foo(self):
        self.createUser('foo', 'foo')
        self.query('GRANT USAGE ON SCHEMA gr_dynout TO foo')
        self.query('GRANT EXECUTE ON SCRIPT gr_dynout.output_columns_as_in_connection_spot4245 TO foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        with self.assertRaisesRegex(Exception, 'insufficient privileges for using connection SPOT4245'):
            foo_conn.query('SELECT gr_dynout.output_columns_as_in_connection_spot4245(1.0)')
        self.query('DROP USER foo CASCADE')

    def test_dynamic_out_from_connection_SPOT4245_for_user_foo_with_view(self):
        self.query('CREATE OR REPLACE VIEW gr_dynout.output_columns_as_in_connection_spot4245_view AS SELECT gr_dynout.output_columns_as_in_connection_spot4245(1.0)')
        self.createUser('foo', 'foo')
        self.query('GRANT SELECT ON gr_dynout.output_columns_as_in_connection_spot4245_view TO foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        foo_conn.query('SELECT * FROM gr_dynout.output_columns_as_in_connection_spot4245_view')
        self.assertEqual(['PASSWORD', 'A', 'B', 'C'], [col[0] for col in foo_conn.cursorDescription()])
        self.query('DROP USER foo CASCADE')
        self.query('DROP VIEW gr_dynout.output_columns_as_in_connection_spot4245_view')


if __name__ == "__main__":
    udf.main()
