#!/usr/bin/env python3

from exasol_python_test_framework import udf


class MetadataRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_meta CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_meta")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_current_schema()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                exa$meta$current_schema
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_database_name()
            RETURNS VARCHAR(300) AS
            run <- function(ctx) {
                exa$meta$database_name
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_database_version()
            RETURNS VARCHAR(20) AS
            run <- function(ctx) {
                exa$meta$database_version
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_script_language()
            EMITS (s1 VARCHAR(300), s2 VARCHAR(300)) AS
            run <- function(ctx) {
                ctx$emit(exa$meta$script_language, 'R')
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_script_name()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                exa$meta$script_name
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_script_schema()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                exa$meta$script_schema
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_current_user()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                exa$meta$current_user
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_scope_user()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                exa$meta$scope_user
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_script_code()
            RETURNS VARCHAR(2000) AS
            run <- function(ctx) {
                exa$meta$script_code
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_session_id()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                as.character(exa$meta$session_id)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_statement_id()
            RETURNS DOUBLE AS
            run <- function(ctx) {
                as.double(exa$meta$statement_id)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_node_id()
            RETURNS DOUBLE AS
            run <- function(ctx) {
                as.double(exa$meta$node_id)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_vm_id()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                as.character(exa$meta$vm_id)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_input_type_scalar()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                exa$meta$input_type
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_meta.get_input_type_set(a DOUBLE)
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                exa$meta$input_type
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_input_column_count_scalar(c1 DOUBLE, c2 VARCHAR(100))
            RETURNS DOUBLE AS
            run <- function(ctx) {
                as.double(exa$meta$input_column_count)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_meta.get_input_column_count_set(c1 DOUBLE, c2 VARCHAR(100))
            RETURNS DOUBLE AS
            run <- function(ctx) {
                as.double(exa$meta$input_column_count)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_input_columns(c1 DOUBLE, c2 VARCHAR(200))
            EMITS (column_id DOUBLE, column_name VARCHAR(200), column_type VARCHAR(100),
                   column_sql_type VARCHAR(100), column_precision DOUBLE, column_scale DOUBLE,
                   column_length DOUBLE) AS
            run <- function(ctx) {
                cols <- exa$meta$input_columns
                for (i in seq_along(cols)) {
                    col <- cols[[i]]
                    ctx$emit(
                        as.double(i),
                        as.character(col$name),
                        as.character(col$type),
                        as.character(col$sql_type),
                        as.double(if (is.null(col$precision)) 0 else col$precision),
                        as.double(if (is.null(col$scale)) 0 else col$scale),
                        as.double(if (is.null(col$length)) 0 else col$length)
                    )
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_output_type_return()
            RETURNS VARCHAR(200) AS
            run <- function(ctx) {
                exa$meta$output_type
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_output_type_emit()
            EMITS (t VARCHAR(200)) AS
            run <- function(ctx) {
                ctx$emit(exa$meta$output_type)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_output_column_count_return()
            RETURNS DOUBLE AS
            run <- function(ctx) {
                as.double(exa$meta$output_column_count)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_output_column_count_emit()
            EMITS (x DOUBLE, y DOUBLE, z DOUBLE) AS
            run <- function(ctx) {
                n <- as.double(exa$meta$output_column_count)
                ctx$emit(n, n, n)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_output_columns()
            EMITS (column_id DOUBLE, column_name VARCHAR(200), column_type VARCHAR(100),
                   column_sql_type VARCHAR(100), column_precision DOUBLE, column_scale DOUBLE,
                   column_length DOUBLE) AS
            run <- function(ctx) {
                cols <- exa$meta$output_columns
                for (i in seq_along(cols)) {
                    col <- cols[[i]]
                    ctx$emit(
                        as.double(i),
                        as.character(col$name),
                        as.character(col$type),
                        as.character(col$sql_type),
                        as.double(if (is.null(col$precision)) 0 else col$precision),
                        as.double(if (is.null(col$scale)) 0 else col$scale),
                        as.double(if (is.null(col$length)) 0 else col$length)
                    )
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_precision_scale_length(n DECIMAL(6, 3), v VARCHAR(10))
            EMITS (precision1 DOUBLE, scale1 DOUBLE, length1 DOUBLE,
                   precision2 DOUBLE, scale2 DOUBLE, length2 DOUBLE) AS
            run <- function(ctx) {
                c1 <- exa$meta$input_columns[[1]]
                c2 <- exa$meta$input_columns[[2]]
                ctx$emit(
                    as.double(if (is.null(c1$precision)) 0 else c1$precision),
                    as.double(if (is.null(c1$scale)) 0 else c1$scale),
                    as.double(if (is.null(c1$length)) 0 else c1$length),
                    as.double(if (is.null(c2$precision)) 0 else c2$precision),
                    as.double(if (is.null(c2$scale)) 0 else c2$scale),
                    as.double(if (is.null(c2$length)) 0 else c2$length)
                )
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_char_length(text CHAR(10))
            EMITS (len1 DOUBLE, len2 DOUBLE, dummy CHAR(20)) AS
            run <- function(ctx) {
                v <- exa$meta$input_columns[[1]]
                w <- exa$meta$output_columns[[3]]
                ctx$emit(
                    as.double(if (is.null(v$length)) 0 else v$length),
                    as.double(if (is.null(w$length)) 0 else w$length),
                    '9876543210'
                )
            };
        """))

        self.query("CLOSE SCHEMA")

    def test_database_name(self):
        rows = self.query("SELECT gr_meta.get_database_name() FROM DUAL")
        self.assertTrue(len(rows[0][0]) > 0)

    def test_database_version(self):
        rows = self.query("SELECT gr_meta.get_database_version() FROM DUAL")
        self.assertTrue(len(rows[0][0]) > 0)

    def test_script_language(self):
        rows = self.query("SELECT gr_meta.get_script_language() FROM DUAL")
        self.assertTrue(rows[0][0].upper().startswith(rows[0][1].upper()))

    def test_script_name(self):
        rows = self.query("SELECT gr_meta.get_script_name() FROM DUAL")
        self.assertRowEqual(('GET_SCRIPT_NAME',), rows[0])

    def test_script_schema(self):
        rows = self.query("SELECT gr_meta.get_script_schema() FROM DUAL")
        self.assertRowEqual(('GR_META',), rows[0])

    def test_script_user(self):
        rows = self.query("SELECT gr_meta.get_current_user() FROM DUAL")
        self.assertRowEqual(('SYS',), rows[0])

    def test_scope_user(self):
        rows = self.query("SELECT gr_meta.get_scope_user() FROM DUAL")
        self.assertRowEqual(('SYS',), rows[0])

    def test_current_schema_null(self):
        rows = self.query("SELECT gr_meta.get_current_schema() FROM DUAL")
        self.assertRowEqual(('NULL',), rows[0])

    def test_current_schema(self):
        rows = self.query("""
            SELECT gr_meta.get_current_schema()
            FROM DUAL
        """)
        self.assertIn(rows[0][0], ('GR_META', 'NULL'))

    # R-only compatibility wrapper for scalar input-column-count coverage.
    def test_input_column_count(self):
        rows = self.query("SELECT gr_meta.get_input_column_count_scalar(1.0, 'x') FROM DUAL")
        self.assertRowsEqual([(2.0,)], rows)

    # R-only compatibility wrapper for emitted output-column-count coverage.
    def test_output_column_count(self):
        rows = self.query("SELECT gr_meta.get_output_column_count_emit() FROM DUAL")
        self.assertRowsEqual([(3.0, 3.0, 3.0)], rows)

    def test_script_code(self):
        rows = self.query("SELECT gr_meta.get_script_code() FROM DUAL")
        self.assertTrue(rows[0][0].lower().find('ctx') >= 0)

    def test_session_id(self):
        rows = self.query("SELECT gr_meta.get_session_id() FROM DUAL")
        self.assertTrue(len(rows[0][0]) > 0)

    def test_statement_id(self):
        rows = self.query("SELECT gr_meta.get_statement_id() FROM DUAL")
        self.assertTrue(rows[0][0] >= 0)

    def test_node_id(self):
        rows = self.query("SELECT gr_meta.get_node_id() FROM DUAL")
        self.assertTrue(rows[0][0] >= 0)

    def test_vm_id(self):
        rows = self.query("SELECT gr_meta.get_vm_id() FROM DUAL")
        self.assertTrue(len(rows[0][0]) > 0)

    def test_input_type_scalar(self):
        rows = self.query("SELECT gr_meta.get_input_type_scalar() FROM DUAL")
        self.assertRowEqual(('SCALAR',), rows[0])

    def test_input_type_set(self):
        rows = self.query("SELECT gr_meta.get_input_type_set(x) FROM (VALUES 1, 2, 3) AS t(x)")
        self.assertRowEqual(('SET',), rows[0])

    def test_input_column_count_scalar(self):
        rows = self.query("SELECT gr_meta.get_input_column_count_scalar(12.3, 'hihihi') FROM DUAL")
        self.assertRowEqual((2,), rows[0])

    def test_input_column_count_set(self):
        rows = self.query("SELECT gr_meta.get_input_column_count_set(x, y) FROM (VALUES (12.3, 'hihihi')) AS t(x, y)")
        self.assertRowEqual((2,), rows[0])

    def test_input_columns(self):
        rows = self.query("SELECT gr_meta.get_input_columns(1.2, '123') FROM DUAL ORDER BY column_id")
        r0 = rows[0]
        r1 = rows[1]
        self.assertEqual(1, r0[0])
        self.assertEqual('C1', r0[1].upper())
        self.assertEqual('DOUBLE', r0[3].upper())
        self.assertEqual(2, r1[0])
        self.assertEqual('C2', r1[1].upper())
        self.assertTrue(r1[3].upper().startswith('VARCHAR(200)'))

    def test_output_type_return(self):
        rows = self.query("SELECT gr_meta.get_output_type_return() FROM DUAL")
        self.assertEqual('RETURN', rows[0][0])

    def test_output_type_emit(self):
        rows = self.query("SELECT gr_meta.get_output_type_emit() FROM DUAL")
        self.assertEqual('EMIT', rows[0][0])

    def test_output_column_count_return(self):
        rows = self.query("SELECT gr_meta.get_output_column_count_return() FROM DUAL")
        self.assertRowEqual((1,), rows[0])

    def test_output_column_count_emit(self):
        rows = self.query("SELECT gr_meta.get_output_column_count_emit() FROM DUAL")
        self.assertRowEqual((3, 3, 3), rows[0])

    def test_output_columns(self):
        rows = self.query("SELECT gr_meta.get_output_columns() FROM DUAL ORDER BY column_id")
        self.assertEqual(7, len(rows))
        self.assertEqual('COLUMN_ID', rows[0][1].upper())
        self.assertEqual('COLUMN_NAME', rows[1][1].upper())
        self.assertEqual('COLUMN_TYPE', rows[2][1].upper())
        self.assertEqual('COLUMN_SQL_TYPE', rows[3][1].upper())
        self.assertEqual('COLUMN_PRECISION', rows[4][1].upper())
        self.assertEqual('COLUMN_SCALE', rows[5][1].upper())
        self.assertEqual('COLUMN_LENGTH', rows[6][1].upper())
        self.assertEqual('DOUBLE', rows[0][3].upper())
        self.assertTrue(rows[1][3].upper().startswith('VARCHAR(200)'))
        self.assertTrue(rows[2][3].upper().startswith('VARCHAR(100)'))
        self.assertTrue(rows[3][3].upper().startswith('VARCHAR(100)'))
        self.assertEqual('DOUBLE', rows[4][3].upper())
        self.assertEqual('DOUBLE', rows[5][3].upper())
        self.assertEqual('DOUBLE', rows[6][3].upper())

    def test_precision_scale_length(self):
        rows = self.query("SELECT gr_meta.get_precision_scale_length(1.234, 'abc') FROM DUAL")
        r = rows[0]
        self.assertEqual(6, r[0])   # precision of DECIMAL(6,3)
        self.assertEqual(3, r[1])   # scale of DECIMAL(6,3)
        self.assertEqual(10, r[5])  # length of VARCHAR(10)

    def test_char_length(self):
        rows = self.query("SELECT gr_meta.get_char_length('hello     ') FROM DUAL")
        r = rows[0]
        self.assertEqual(10, r[0])   # CHAR(10) input length
        self.assertEqual(20, r[1])   # CHAR(20) output length
        self.assertEqual('9876543210', r[2].rstrip())


if __name__ == "__main__":
    udf.main()
