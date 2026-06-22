#!/usr/bin/env python3

from exasol_python_test_framework import udf


class DynamicOutputRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_dynout CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA gr_dynout_data CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_dynout")
        self.query("CREATE SCHEMA gr_dynout_data")
        self.query("CREATE TABLE gr_dynout_data.small(x VARCHAR(2000), y DOUBLE)")
        self.query("INSERT INTO gr_dynout_data.small VALUES ('abc', 2.2)")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_simple_set(a DOUBLE)
            EMITS (...) AS
            run <- function(ctx) {
                ctx$emit(1)
            }

            default_output_columns <- function() {
                'x DOUBLE'
            }
            /
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynout.varemit_emit_input(...)
            EMITS (...) AS
            run <- function(ctx) {
                out <- list()
                for (i in 1:exa$meta$input_column_count) {
                    out[[i]] <- ctx[[i]]()
                }
                do.call(ctx$emit, out)
            }

            default_output_columns <- function() {
                'a VARCHAR(2000), b DOUBLE'
            }
            /
        """))

    def test_default_dynamic_output_columns(self):
        rows = self.query("""
            SELECT gr_dynout.varemit_simple_set(1.2)
            FROM DUAL
        """)
        self.assertRowsEqual([(1.0,)], rows)

    def test_copy_input_relation(self):
        rows = self.query("""
            SELECT gr_dynout.varemit_emit_input(x, y)
            FROM gr_dynout_data.small
        """)
        self.assertRowsEqual([('abc', 2.2)], rows)


if __name__ == "__main__":
    udf.main()
