#!/usr/bin/env python3

from exasol_python_test_framework import udf


class BasicRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_basic CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_basic_data CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_basic")
        self.query("CREATE SCHEMA gr_basic_data")

        self.query("CREATE TABLE gr_basic_data.empty_table(c INT)")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_basic.basic_range(n INTEGER)
            EMITS (n INTEGER) AS
            run <- function(ctx) {
                if (!is.null(ctx$n)) {
                    for (i in 0:(ctx$n - 1)) {
                        ctx$emit(i)
                    }
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_basic.basic_sum(x INTEGER)
            RETURNS INTEGER AS
            run <- function(ctx) {
                if (is.null(ctx$x)) {
                    return(as.integer(0))
                }
                s <- as.integer(ctx$x)
                while (ctx$next_row()) {
                    if (!is.null(ctx$x)) {
                        s <- s + as.integer(ctx$x)
                    }
                }
                s
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_basic.basic_emit_two_ints()
            EMITS (i INTEGER, j INTEGER) AS
            run <- function(ctx) {
                ctx$emit(1L, 2L)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_basic.set_returns_has_empty_input(a DOUBLE)
            RETURNS BOOLEAN AS
            run <- function(ctx) {
                is.null(ctx$a)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_basic.set_emits_has_empty_input(a DOUBLE)
            EMITS (x DOUBLE, y VARCHAR(10)) AS
            run <- function(ctx) {
                if (is.null(ctx$a)) {
                    ctx$emit(1, '1')
                } else {
                    ctx$emit(2, '2')
                }
            };
        """))

    def test_basic_scalar_emits(self):
        rows = self.query("""
            SELECT gr_basic.basic_range(3)
            FROM DUAL
        """)
        self.assertRowsEqual([(0,), (1,), (2,)], sorted(rows))

    def test_basic_set_returns(self):
        rows = self.query("""
            SELECT gr_basic.basic_sum(3)
            FROM DUAL
        """)
        self.assertRowsEqual([(3,)], rows)

    def test_emit_two_ints(self):
        rows = self.query("""
            SELECT gr_basic.basic_emit_two_ints()
            FROM DUAL
        """)
        self.assertRowsEqual([(1, 2)], rows)

    def test_set_with_empty_input(self):
        rows = self.query("""
            SELECT gr_basic.set_returns_has_empty_input(c)
            FROM gr_basic_data.empty_table
        """)
        self.assertRowsEqual([(None,)], rows)


if __name__ == "__main__":
    udf.main()
