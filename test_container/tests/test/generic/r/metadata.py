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
                exa$meta$script_schema
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
            CREATE OR REPLACE R SCALAR SCRIPT gr_meta.get_output_column_count_emit()
            EMITS (x DOUBLE, y DOUBLE, z DOUBLE) AS
            run <- function(ctx) {
                n <- as.double(exa$meta$output_column_count)
                ctx$emit(n, n, n)
            };
        """))

    def test_current_schema(self):
        rows = self.query("""
            SELECT gr_meta.get_current_schema()
            FROM DUAL
        """)
        self.assertRowsEqual([('GR_META',)], rows)

    def test_input_column_count(self):
        rows = self.query("""
            SELECT gr_meta.get_input_column_count_scalar(1.0, 'x')
            FROM DUAL
        """)
        self.assertRowsEqual([(2.0,)], rows)

    def test_output_column_count(self):
        rows = self.query("""
            SELECT gr_meta.get_output_column_count_emit()
            FROM DUAL
        """)
        self.assertRowsEqual([(3.0, 3.0, 3.0)], rows)


if __name__ == "__main__":
    udf.main()
