#!/usr/bin/env python3

import sys

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import SkipTest, useData


class Vectorsize(udf.TestCase):
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

    def test_vectorsize_5000(self):
        rows = self.query("""
            SELECT LENGTH(gr_vec.vectorsize(5000, 1.0))
            FROM DUAL
        """)
        expected_len = len(''.join(str(i) for i in range(5000)))
        self.assertRowsEqual([(expected_len,)], rows)

    data = [
        (10,),
        (30,),
        (100,),
        (300,),
        (1000,),
        (3000,),
        (10000,),
        (30000,),
        (100000,),
        (200000,),
        (351850,),
    ]

    @useData(data)
    def test_vectorsize(self, size):
        limits = {
            'lua': 100000,
            'python3': 8000,
            'r': 3000,
            'java': 3000,
        }
        if size > limits.get('r', sys.maxsize):
            raise SkipTest('test is to slow')
        rows = self.query("""
            SELECT LENGTH(gr_vec.vectorsize(%d, 1.0))
            FROM DUAL
        """ % size)
        expected_len = len(''.join(str(i) for i in range(size)))
        self.assertRowsEqual([(expected_len,)], rows)

    data = [
        (10, 10, 10),
        (100, 100, 100),
        (1000, 100, 100),
        (10000, 100, 100),
        (100000, 100, 100),
        (351850, 100, 100),
        (100, 10, 100000),
        (100, 100, 10000),
        (100, 1000, 1000),
        (100, 10000, 100),
        (100, 100000, 10),
    ]

    @useData(data)
    def test_vectorsize_set(self, a, b, c):
        if max(a, b, c) > 3000:
            raise SkipTest('test is to slow')
        rows = self.query("""
            SELECT COUNT(*)
            FROM (
                SELECT gr_vec.vectorsize_set(%d, %d, n)
                FROM (
                    SELECT gr_vec.basic_range(%d)
                    FROM DUAL
                )
            )
        """ % (a, b, c))
        self.assertRowsEqual([(b * c,)], rows)


if __name__ == "__main__":
    udf.main()
