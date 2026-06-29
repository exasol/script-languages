#!/usr/bin/env python3

from exasol_python_test_framework import udf


class GenericTypesRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_types CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_types")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_boolean(x BOOLEAN)
            RETURNS BOOLEAN AS
            run <- function(ctx) {
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_integer(x INTEGER)
            RETURNS INTEGER AS
            run <- function(ctx) {
                if (is.null(ctx$x)) {
                    return(NULL)
                }
                as.integer(ctx$x)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_double(x DOUBLE)
            RETURNS DOUBLE AS
            run <- function(ctx) {
                if (is.null(ctx$x)) {
                    return(NULL)
                }
                as.double(ctx$x)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_varchar10(x VARCHAR(10))
            RETURNS VARCHAR(10) AS
            run <- function(ctx) {
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_char1(x CHAR(1))
            RETURNS CHAR(1) AS
            run <- function(ctx) {
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_char10(x CHAR(10))
            RETURNS CHAR(10) AS
            run <- function(ctx) {
                if (is.null(ctx$x) || is.na(ctx$x)) return(NULL)
                len <- nchar(ctx$x, type = 'chars', allowNA = TRUE, keepNA = TRUE)
                if (is.na(len) || len != 10L) return(NULL)
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_date(x DATE)
            RETURNS DATE AS
            run <- function(ctx) {
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_timestamp(x TIMESTAMP)
            RETURNS TIMESTAMP AS
            run <- function(ctx) {
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_decimal_36_0(x DECIMAL(36, 0))
            RETURNS DECIMAL(36, 0) AS
            run <- function(ctx) {
                if (is.null(ctx$x)) return(NULL)
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_decimal_36_36(x DECIMAL(36, 36))
            RETURNS DECIMAL(36, 36) AS
            run <- function(ctx) {
                if (is.null(ctx$x)) return(NULL)
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.run_func_is_empty()
            RETURNS DOUBLE AS
            run <- function(ctx) {
                NULL
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.bottleneck_varchar10(i VARCHAR(20))
            RETURNS VARCHAR(10) AS
            run <- function(ctx) {
                ctx$i
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.bottleneck_char10(i VARCHAR(20))
            RETURNS CHAR(10) AS
            run <- function(ctx) {
                ctx$i
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.bottleneck_decimal5(i DECIMAL(20, 0))
            RETURNS DECIMAL(5, 0) AS
            run <- function(ctx) {
                ctx$i
            };
        """))

    def test_echo_boolean(self):
        rows = self.query("""
            SELECT
                gr_types.echo_boolean(NULL) IS NULL,
                gr_types.echo_boolean(TRUE) = TRUE,
                gr_types.echo_boolean(FALSE) = FALSE
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True)], rows)

    # R-only smoke test kept for a minimal integer echo path.
    def test_echo_integer(self):
        rows = self.query("""
            SELECT
                gr_types.echo_integer(NULL) IS NULL,
                gr_types.echo_integer(1) = 1,
                gr_types.echo_integer(-1) = -1
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True)], rows)

    def test_echo_integer_basic(self):
        rows = self.query("""
            SELECT
                gr_types.echo_integer(NULL) IS NULL,
                gr_types.echo_integer(-1) = -1,
                gr_types.echo_integer(0) = 0,
                gr_types.echo_integer(1) = 1
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True, True)], rows)

    def test_echo_double(self):
        rows = self.query("""
            SELECT
                gr_types.echo_double(NULL) IS NULL,
                gr_types.echo_double(CAST(1.5 AS DOUBLE)) = CAST(1.5 AS DOUBLE),
                gr_types.echo_double(0) = 0.0,
                gr_types.echo_double(0.0) = 0.0
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True, True)], rows)

    # R-only mixed-type smoke test for double and varchar together.
    def test_echo_double_and_varchar(self):
        rows = self.query("""
            SELECT
                gr_types.echo_double(1.5) = 1.5,
                gr_types.echo_varchar10('abc') = 'abc'
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True)], rows)

    def test_echo_char1(self):
        rows = self.query("""
            SELECT
                gr_types.echo_char1(NULL) IS NULL,
                gr_types.echo_char1('a') = 'a'
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True)], rows)

    def test_echo_char10(self):
        rows = self.query("""
            SELECT
                gr_types.echo_char10(NULL) IS NULL,
                gr_types.echo_char10('ab') = 'ab        '
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True)], rows)

    def test_echo_date(self):
        rows = self.query("""
            SELECT
                gr_types.echo_date(NULL) IS NULL,
                gr_types.echo_date(CURRENT_DATE()) = CURRENT_DATE()
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True)], rows)

    def test_echo_timestamp(self):
        rows = self.query("""
            SELECT gr_types.echo_timestamp(NULL) IS NULL
            FROM DUAL
        """)
        self.assertRowsEqual([(True,)], rows)

        rows = self.query("""
            SELECT gr_types.echo_timestamp('2017-08-01 13:13:50.910') = '2017-08-01 13:13:50.910'
            FROM DUAL
        """)
        self.assertRowsEqual([(True,)], rows)

    def test_echo_decimal_36_0_basic(self):
        rows = self.query("""
            SELECT
                gr_types.echo_decimal_36_0(NULL) IS NULL,
                gr_types.echo_decimal_36_0(0) = 0,
                gr_types.echo_decimal_36_0(0.0) = 0.0
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True)], rows)

    def test_echo_decimal_36_36_basic(self):
        rows = self.query("""
            SELECT
                gr_types.echo_decimal_36_36(NULL) IS NULL,
                gr_types.echo_decimal_36_36(0) = 0,
                gr_types.echo_decimal_36_36(0.0) = 0.0
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True)], rows)

    def test_echo_varchar10(self):
        rows = self.query("""
            SELECT
                gr_types.echo_varchar10(NULL) IS NULL,
                gr_types.echo_varchar10(' ') = ' ',
                gr_types.echo_varchar10('a') = 'a',
                gr_types.echo_varchar10('a ') = 'a ',
                gr_types.echo_varchar10(' a ') = ' a '
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True, True, True)], rows)

    def test_run_func_is_empty(self):
        rows = self.query("""
            SELECT gr_types.run_func_is_empty() IS NULL
            FROM DUAL
        """)
        self.assertRowsEqual([(True,)], rows)

    def test_varchar10(self):
        for i in [0, 1, 5, 10]:
            rows = self.query("""
                SELECT gr_types.bottleneck_varchar10('%s')
                FROM DUAL
            """ % ('x' * i))
            self.assertEqual('x' * i if i > 0 else None, rows[0][0])
        with self.assertRaises(Exception):
            self.query("""
                SELECT gr_types.bottleneck_varchar10('%s')
                FROM DUAL
            """ % ('x' * 11))

    def test_char10(self):
        for i in [0, 1, 5, 10]:
            rows = self.query("""
                SELECT gr_types.bottleneck_char10('%s')
                FROM DUAL
            """ % ('x' * i))
            self.assertEqual(('x' * i + ' ' * 10)[:10] if i > 0 else None, rows[0][0])
        with self.assertRaises(Exception):
            self.query("""
                SELECT gr_types.bottleneck_char10('%s')
                FROM DUAL
            """ % ('x' * 11))

    def test_decimal5(self):
        for i in [3, 4]:
            rows = self.query("""
                SELECT gr_types.bottleneck_decimal5(%d)
                FROM DUAL
            """ % (10 ** i))
            self.assertEqual(10 ** i, rows[0][0])
        with self.assertRaises(Exception):
            self.query("""
                SELECT gr_types.bottleneck_decimal5(%d)
                FROM DUAL
            """ % (10 ** 5))


    # Generic parity note:
    # The following tests from test/generic/generic_types.py are intentionally
    # not implemented in R due to DWA-13784 (R runtime does not preserve these
    # boundary/precision numeric values reliably):
    # - test_echo_integer_limits
    # - test_echo_decimal_36_0_limits
    # - test_echo_decimal_36_36_limits


if __name__ == "__main__":
    udf.main()
