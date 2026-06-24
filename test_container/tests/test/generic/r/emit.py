#!/usr/bin/env python3

import datetime

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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_emit.line_3i_2o(x DOUBLE, y DOUBLE, z DOUBLE)
            EMITS (z1 DOUBLE, z2 DOUBLE) AS
            run <- function(ctx) {
                ctx$emit(ctx$x + ctx$y, 3000)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_emit.dob_1i_1o(x DOUBLE)
            EMITS (y DOUBLE) AS
            run <- function(ctx) {
                ctx$emit(ctx$x)
                ctx$emit(ctx$x)
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

    def test_iomatch_3i_2o(self):
        rows = self.query("""
            SELECT x * 2, gr_emit.line_3i_2o(x, id, id), x * 3
            FROM gr_emit_data.t
        """)
        self.assertRowsEqual(sorted([(2, 101, 3000, 3), (6, 203, 3000, 9), (4, 102, 3000, 6)]), sorted(rows))

    def test_iomatch_dob_1i_1o(self):
        rows = self.query("""
            SELECT x * 2, gr_emit.dob_1i_1o(x), x * 3
            FROM gr_emit_data.t
        """)
        self.assertRowsEqual(sorted([(2,1,3),(2,1,3),(6,3,9),(6,3,9),(4,2,6),(4,2,6)]), sorted(rows))

    def test_col_names(self):
        self.query("""
            CREATE OR REPLACE TABLE gr_emit_data.foo AS
            SELECT x * 2 a, gr_emit.line_3i_2o(x, id, id), x * 3 b
            FROM gr_emit_data.t
        """)
        rows = self.query("DESCRIBE gr_emit_data.foo")
        self.assertEqual('A', rows[0][0])
        self.assertEqual('Z1', rows[1][0])
        self.assertEqual('Z2', rows[2][0])
        self.assertEqual('B', rows[3][0])

    def test_boolean(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x BOOLEAN)")
        self.query("INSERT INTO gr_emit_data.dt VALUES FALSE")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([(False, 0)], rows)

    def test_double(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x DOUBLE)")
        self.query("INSERT INTO gr_emit_data.dt VALUES 32768e100")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([(3.2768e+104, 0)], rows)

    def test_dec_32bit(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x DECIMAL(9,0))")
        self.query("INSERT INTO gr_emit_data.dt VALUES 32768")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([(32768, 0)], rows)

    def test_dec_64bit(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x DECIMAL(18,0))")
        self.query("INSERT INTO gr_emit_data.dt VALUES 32768")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([(32768, 0)], rows)

    def test_dec_128bit(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x DECIMAL(36,0))")
        self.query("INSERT INTO gr_emit_data.dt VALUES 32768")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([(32768, 0)], rows)

    def test_timestamp(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x TIMESTAMP)")
        self.query("INSERT INTO gr_emit_data.dt VALUES '2010-01-01 23:33:33'")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([(datetime.datetime(2010, 1, 1, 23, 33, 33), 0)], rows)

    def test_date(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x DATE)")
        self.query("INSERT INTO gr_emit_data.dt VALUES '2010-01-01'")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([(datetime.date(2010, 1, 1), 0)], rows)

    def test_varchar_utf8(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x VARCHAR(3000) UTF8)")
        self.query("INSERT INTO gr_emit_data.dt VALUES REPEAT(5, 300)")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([('5' * 300, 0)], rows)

    def test_varchar_ascii(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x VARCHAR(3000) ASCII)")
        self.query("INSERT INTO gr_emit_data.dt VALUES REPEAT(5, 300)")
        rows = self.query("SELECT x, gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([('5' * 300, 0)], rows)

    def test_boolean_null(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x BOOLEAN)")
        self.query("INSERT INTO gr_emit_data.dt VALUES NULL")
        rows = self.query("SELECT gr_emit.line_1i_1o(0) FROM gr_emit_data.dt")
        self.assertRowsEqual([(0,)], rows)

    def test_double_null(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x DOUBLE)")
        self.query("INSERT INTO gr_emit_data.dt VALUES NULL")
        rows = self.query("SELECT x IS NULL FROM gr_emit_data.dt")
        self.assertRowsEqual([(True,)], rows)

    def test_varchar_null(self):
        self.query("CREATE OR REPLACE TABLE gr_emit_data.dt(x VARCHAR(100))")
        self.query("INSERT INTO gr_emit_data.dt VALUES NULL")
        rows = self.query("SELECT x IS NULL FROM gr_emit_data.dt")
        self.assertRowsEqual([(True,)], rows)


if __name__ == "__main__":
    udf.main()
