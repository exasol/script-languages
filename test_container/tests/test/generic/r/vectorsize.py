#!/usr/bin/env python3

from exasol_python_test_framework import udf


class VectorSizeRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_vec CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_vec")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_vec.basic_range(n INTEGER)
            EMITS (n INTEGER) AS
            run <- function(ctx) {
                if (is.null(ctx$n) || ctx$n <= 0) {
                    return(NULL)
                }
                for (i in 0:(ctx$n - 1)) {
                    ctx$emit(i)
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_vec.vectorsize(length INT, dummy DOUBLE)
            RETURNS VARCHAR(2000000) AS
            run <- function(ctx) {
                paste(0:(as.integer(ctx$length) - 1), collapse = '')
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_vec.vectorsize_set(length INT, n INT, dummy DOUBLE)
            EMITS (o VARCHAR(2000000)) AS
            run <- function(ctx) {
                val <- paste(0:(as.integer(ctx$length) - 1), collapse = '')
                repeat {
                    for (i in 1:as.integer(ctx$n)) {
                        ctx$emit(val)
                    }
                    if (!ctx$next_row()) {
                        break
                    }
                }
            };
        """))

    def test_vectorsize(self):
        rows = self.query("""
            SELECT gr_vec.vectorsize(10, 1.0)
            FROM DUAL
        """)
        self.assertRowsEqual([('0123456789',)], rows)

    def test_vectorsize_100(self):
        rows = self.query("""
            SELECT LENGTH(gr_vec.vectorsize(100, 1.0))
            FROM DUAL
        """)
        expected_len = len(''.join(str(i) for i in range(100)))
        self.assertRowsEqual([(expected_len,)], rows)

    def test_vectorsize_1000(self):
        rows = self.query("""
            SELECT LENGTH(gr_vec.vectorsize(1000, 1.0))
            FROM DUAL
        """)
        expected_len = len(''.join(str(i) for i in range(1000)))
        self.assertRowsEqual([(expected_len,)], rows)

    def test_vectorsize_3000(self):
        rows = self.query("""
            SELECT LENGTH(gr_vec.vectorsize(3000, 1.0))
            FROM DUAL
        """)
        expected_len = len(''.join(str(i) for i in range(3000)))
        self.assertRowsEqual([(expected_len,)], rows)

    def test_vectorsize_5000(self):
        rows = self.query("""
            SELECT LENGTH(gr_vec.vectorsize(5000, 1.0))
            FROM DUAL
        """)
        expected_len = len(''.join(str(i) for i in range(5000)))
        self.assertRowsEqual([(expected_len,)], rows)

    def test_vectorsize_set(self):
        rows = self.query("""
            SELECT COUNT(*)
            FROM (
                SELECT gr_vec.vectorsize_set(5, 3, n)
                FROM (
                    SELECT gr_vec.basic_range(2)
                    FROM DUAL
                )
            )
        """)
        self.assertRowsEqual([(6,)], rows)

    def test_vectorsize_set_10_10(self):
        rows = self.query("""
            SELECT COUNT(*)
            FROM (
                SELECT gr_vec.vectorsize_set(10, 10, n)
                FROM (
                    SELECT gr_vec.basic_range(3)
                    FROM DUAL
                )
            )
        """)
        self.assertRowsEqual([(30,)], rows)

    def test_vectorsize_set_100_100(self):
        rows = self.query("""
            SELECT COUNT(*)
            FROM (
                SELECT gr_vec.vectorsize_set(100, 100, n)
                FROM (
                    SELECT gr_vec.basic_range(5)
                    FROM DUAL
                )
            )
        """)
        self.assertRowsEqual([(500,)], rows)


if __name__ == "__main__":
    udf.main()
