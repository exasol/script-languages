#!/usr/bin/env python3

from exasol_python_test_framework import udf


class DynamicOutputRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_dynout CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_dynout_data CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_dynout")
        self.query("CREATE SCHEMA gr_dynout_data")
        self.query("CREATE TABLE gr_dynout_data.small(x VARCHAR(2000), y DOUBLE)")
        self.query("INSERT INTO gr_dynout_data.small VALUES ('abc', 2.2)")

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

        self.query("CREATE TABLE gr_dynout_data.target (a INT, b DOUBLE, c VARCHAR(100))")
        self.query("CREATE TABLE gr_dynout_data.groupt(id INT, n DOUBLE, v VARCHAR(999))")
        self.query("INSERT INTO gr_dynout_data.groupt VALUES (1, 1, 'aa'), (1, 2, 'ab'), (2, 2, 'ba')")

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

    def test_error_emit_missing(self):
        with self.assertRaisesRegex(Exception, 'The script has dynamic return arguments'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1)")

    def test_error_empty_emit(self):
        with self.assertRaisesRegex(Exception, 'Empty return argument definition is not allowed'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1) EMITS ()")

    def test_error_wrong_emit(self):
        with self.assertRaisesRegex(Exception, 'syntax error'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1) EMITS (int)")

    def test_error_redundant_name(self):
        with self.assertRaisesRegex(Exception, 'Return argument A is declared more than once'):
            self.query("SELECT gr_dynout.varemit_generic_emit(1) EMITS (a INT, b INT, a INT)")

    def test_error_non_var_emit(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition'):
            self.query("SELECT gr_dynout.varemit_non_var_emit(1) EMITS (a DOUBLE)")

    def test_error_returns_not_supported(self):
        with self.assertRaisesRegex(Exception, 'The script has a static return argument definition'):
            self.query("SELECT gr_dynout.varemit_simple_returns(1) EMITS (a INT)")

    def test_insert_basic(self):
        self.query("DELETE FROM gr_dynout_data.target")
        self.query("""
            INSERT INTO gr_dynout_data.target
            SELECT gr_dynout.varemit_emit_input(1, CAST(1.1 AS DOUBLE), 'a')
        """)
        rows = self.query("SELECT * FROM gr_dynout_data.target")
        self.assertRowEqual((1, 1.1, 'a'), rows[0])
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


if __name__ == "__main__":
    udf.main()
