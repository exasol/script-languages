#!/usr/bin/env python3

from exasol_python_test_framework import udf


class DynamicInputRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_dynin CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_dynin_data CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_dynin")
        self.query("CREATE SCHEMA gr_dynin_data")
        self.query("CREATE TABLE gr_dynin_data.small(x VARCHAR(2000), y DOUBLE)")
        self.query("INSERT INTO gr_dynin_data.small VALUES ('Some string ... and some more', 2.2)")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_dynin.basic_scalar_emit(...)
            EMITS (v VARCHAR(2000)) AS
            run <- function(ctx) {
                for (i in 1:exa$meta$input_column_count) {
                    ctx$emit(as.character(ctx[[i]]()))
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_dynin.basic_scalar_return(...)
            RETURNS VARCHAR(2000) AS
            run <- function(ctx) {
                as.character(ctx[[exa$meta$input_column_count]]())
            };
        """))

    def test_basic_scalar_emit_constants(self):
        rows = self.query("""
            SELECT gr_dynin.basic_scalar_emit('abc', CAST(99 AS DOUBLE))
            FROM DUAL
        """)
        self.assertRowsEqual([('abc',), ('99',)], rows)

    def test_basic_scalar_emit_table(self):
        rows = self.query("""
            SELECT gr_dynin.basic_scalar_emit(x, y)
            FROM gr_dynin_data.small
        """)
        self.assertRowsEqual([('Some string ... and some more',), ('2.2',)], rows)

    def test_basic_scalar_return(self):
        rows = self.query("""
            SELECT gr_dynin.basic_scalar_return(x, y, x, y)
            FROM gr_dynin_data.small
        """)
        self.assertRowsEqual([('2.2',)], rows)


if __name__ == "__main__":
    udf.main()
