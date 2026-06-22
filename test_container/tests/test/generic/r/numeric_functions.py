#!/usr/bin/env python3

from exasol_python_test_framework import udf


class NumericFunctionsRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_num CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_num")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_num.add_two_doubles(x DOUBLE, y DOUBLE)
            RETURNS DOUBLE AS
            run <- function(ctx) {
                if (is.null(ctx$x) || is.null(ctx$y)) {
                    return(NULL)
                }
                ctx$x + ctx$y
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_num.add_three_doubles(x DOUBLE, y DOUBLE, z DOUBLE)
            RETURNS DOUBLE AS
            run <- function(ctx) {
                if (is.null(ctx$x) || is.null(ctx$y) || is.null(ctx$z)) {
                    return(NULL)
                }
                ctx$x + ctx$y + ctx$z
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_num.pi()
            RETURNS DOUBLE AS
            run <- function(ctx) {
                pi
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_num.split_integer_into_digits(x INTEGER)
            EMITS (y INTEGER) AS
            run <- function(ctx) {
                if (is.null(ctx$x)) {
                    return(NULL)
                }
                n <- abs(as.integer(ctx$x))
                if (n == 0L) {
                    ctx$emit(0L)
                    return(NULL)
                }
                while (n > 0L) {
                    ctx$emit(as.integer(n %% 10L))
                    n <- as.integer(n %/% 10L)
                }
            };
        """))

    def test_pi(self):
        rows = self.query("SELECT gr_num.pi() FROM DUAL")
        self.assertAlmostEqual(3.1415926535, rows[0][0])

    def test_udf_with_two_doubles(self):
        rows = self.query("""
            SELECT
                gr_num.add_two_doubles(NULL, NULL) IS NULL,
                gr_num.add_two_doubles(NULL,    0) IS NULL,
                gr_num.add_two_doubles(   0, NULL) IS NULL,
                gr_num.add_two_doubles(0, 0) = 0,
                gr_num.add_two_doubles(1, 0) = 1,
                gr_num.add_two_doubles(0, 2) = 2,
                gr_num.add_two_doubles(2, 3) = 5
            FROM DUAL
        """)
        self.assertRowsEqual([tuple([True] * 7)], rows)

    def test_udf_with_three_doubles_part1(self):
        rows = self.query("""
            SELECT
                gr_num.add_three_doubles(NULL, NULL, NULL) IS NULL,
                gr_num.add_three_doubles(NULL, NULL,    0) IS NULL,
                gr_num.add_three_doubles(NULL,    0, NULL) IS NULL,
                gr_num.add_three_doubles(   0, NULL, NULL) IS NULL,
                gr_num.add_three_doubles(NULL,    0,    0) IS NULL
            FROM DUAL
        """)
        self.assertRowsEqual([tuple([True] * 5)], rows)

    def test_udf_with_three_doubles_part2(self):
        rows = self.query("""
            SELECT
                gr_num.add_three_doubles(   0, NULL,    0) IS NULL,
                gr_num.add_three_doubles(   0,    0, NULL) IS NULL,
                gr_num.add_three_doubles(0, 0, 0) = 0,
                gr_num.add_three_doubles(1, 0, 0) = 1,
                gr_num.add_three_doubles(0, 2, 0) = 2
            FROM DUAL
        """)
        self.assertRowsEqual([tuple([True] * 5)], rows)

    def test_udf_with_three_doubles_part3(self):
        rows = self.query("""
            SELECT
                gr_num.add_three_doubles(0, 0, 3) = 3,
                gr_num.add_three_doubles(1, 2, 0) = 3,
                gr_num.add_three_doubles(1, 0, 3) = 4,
                gr_num.add_three_doubles(0, 2, 3) = 5,
                gr_num.add_three_doubles(1, 2, 3) = 6
            FROM DUAL
        """)
        self.assertRowsEqual([tuple([True] * 5)], rows)

    def test_add_functions(self):
        rows = self.query("""
            SELECT
                gr_num.add_two_doubles(2, 3) = 5,
                gr_num.add_three_doubles(1, 2, 3) = 6
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True)], rows)

    def test_digit_split(self):
        rows = self.query("""
            SELECT gr_num.split_integer_into_digits(123)
            FROM DUAL
        """)
        self.assertRowsEqual([(3,), (2,), (1,)], rows)

    def test_right_number_of_emitted_rows(self):
        rows = self.query("""
            SELECT gr_num.split_integer_into_digits(12345)
            FROM DUAL
        """)
        self.assertEqual(5, len(rows))


if __name__ == "__main__":
    udf.main()
