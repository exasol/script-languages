#!/usr/bin/env python3

from exasol_python_test_framework import udf


class EmitRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_emit CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_emit_data CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_emit")
        self.query("CREATE SCHEMA gr_emit_data")
        self.query("CREATE TABLE gr_emit_data.t(id DOUBLE, x DOUBLE)")
        self.query("INSERT INTO gr_emit_data.t VALUES (100,1),(100,2),(200,3)")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_emit.line_1i_1o(x DOUBLE)
            EMITS (y DOUBLE) AS
            run <- function(ctx) {
                ctx$emit(ctx$x)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_emit.line_1i_2o(x DOUBLE)
            EMITS (y DOUBLE, z DOUBLE) AS
            run <- function(ctx) {
                ctx$emit(ctx$x, ctx$x)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_emit.line_2i_1o(x DOUBLE, y DOUBLE)
            EMITS (z DOUBLE) AS
            run <- function(ctx) {
                ctx$emit(ctx$x + ctx$y)
            };
        """))

    def test_iomatch_1i_1o(self):
        rows = self.query("""
            SELECT x * 2, gr_emit.line_1i_1o(x), x * 3
            FROM gr_emit_data.t
        """)
        self.assertRowsEqual(sorted([(2, 1, 3), (4, 2, 6), (6, 3, 9)]), sorted(rows))

    def test_iomatch_1i_2o(self):
        rows = self.query("""
            SELECT x * 2, gr_emit.line_1i_2o(x), x * 3
            FROM gr_emit_data.t
        """)
        self.assertRowsEqual(sorted([(2, 1, 1, 3), (4, 2, 2, 6), (6, 3, 3, 9)]), sorted(rows))

    def test_iomatch_2i_1o(self):
        rows = self.query("""
            SELECT x * 2, gr_emit.line_2i_1o(x, id), x * 3
            FROM gr_emit_data.t
        """)
        self.assertRowsEqual(sorted([(2, 101, 3), (4, 102, 6), (6, 203, 9)]), sorted(rows))


if __name__ == "__main__":
    udf.main()
