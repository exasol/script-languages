#!/usr/bin/env python3

from exasol_python_test_framework import udf


class CombinationsRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_combi CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_combi_data CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_combi")
        self.query("CREATE SCHEMA gr_combi_data")
        self.query("CREATE TABLE gr_combi_data.small(x DOUBLE, y DOUBLE)")
        self.query("INSERT INTO gr_combi_data.small VALUES (0.1, 0.2), (0.2, 0.1)")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_combi.scalar_returns(x DOUBLE, y DOUBLE)
            RETURNS DOUBLE AS
            run <- function(ctx) {
                ctx$x + ctx$y
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_combi.scalar_emits(x DOUBLE, y DOUBLE)
            EMITS (x DOUBLE, y DOUBLE) AS
            run <- function(ctx) {
                start <- as.integer(ctx$x)
                stop <- as.integer(ctx$y)
                if (start <= stop) {
                    for (i in start:stop) {
                        ctx$emit(as.double(i), as.double(i * i))
                    }
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_combi.set_returns(x DOUBLE, y DOUBLE)
            RETURNS DOUBLE AS
            run <- function(ctx) {
                acc <- as.double(ctx$x + ctx$y)
                while (ctx$next_row()) {
                    acc <- acc + as.double(ctx$x + ctx$y)
                }
                acc
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_combi.set_emits(x DOUBLE, y DOUBLE)
            EMITS (x DOUBLE, y DOUBLE) AS
            run <- function(ctx) {
                repeat {
                    ctx$emit(ctx$y, ctx$x)
                    if (!ctx$next_row()) {
                        break
                    }
                }
            };
        """))

    def test_scalar_returns(self):
        rows = self.query("""
            SELECT ROUND(gr_combi.scalar_returns(x, y) / 2)
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(0,), (0,)], rows)

    def test_set_returns(self):
        rows = self.query("""
            SELECT gr_combi.set_returns(x, y)
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(0.6,)], rows)

    def test_set_emits(self):
        rows = self.query("""
            SELECT gr_combi.set_emits(x * 10, y * 10)
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(2.0, 1.0), (1.0, 2.0)], rows)


if __name__ == "__main__":
    udf.main()
