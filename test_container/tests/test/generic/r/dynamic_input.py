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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynin.basic_set_emit(...)
            EMITS (v VARCHAR(2000)) AS
            run <- function(ctx) {
                repeat {
                    for (i in 1:exa$meta$input_column_count) {
                        ctx$emit(as.character(ctx[[i]]()))
                    }
                    if (!ctx$next_row()) break
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_dynin.basic_set_return(...)
            RETURNS VARCHAR(2000) AS
            run <- function(ctx) {
                result <- 'result: '
                repeat {
                    for (i in 1:exa$meta$input_column_count) {
                        result <- paste0(result, as.character(ctx[[i]]()), ' , ')
                    }
                    if (!ctx$next_row()) break
                }
                result
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_dynin.metadata_scalar_return(...)
            RETURNS VARCHAR(2000) AS
            run <- function(ctx) {
                as.character(exa$meta$input_column_count)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_dynin.metadata_scalar_emit(...)
            EMITS (v VARCHAR(2000)) AS
            run <- function(ctx) {
                ctx$emit(as.character(exa$meta$input_column_count))
                for (i in 1:exa$meta$input_column_count) {
                    col <- exa$meta$input_columns[[i]]
                    ctx$emit(as.character(col$name))
                    ctx$emit(as.character(col$type))
                    ctx$emit(as.character(col$sql_type))
                    ctx$emit(as.character(col$precision))
                    ctx$emit(as.character(col$scale))
                    ctx$emit(as.character(col$length))
                }
            };
        """))

        self.query("CREATE TABLE gr_dynin_data.groupt(id INT, n DOUBLE, v VARCHAR(999))")
        self.query("INSERT INTO gr_dynin_data.groupt VALUES (1, 1, 'aa'), (1, 2, 'ab'), (2, 2, 'ba')")

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

    def test_meta_scalar_return(self):
        rows = self.query("""
            SELECT gr_dynin.metadata_scalar_return('abc', CAST(99 AS DOUBLE))
            FROM DUAL
        """)
        self.assertRowsEqual([('2',)], rows)

    def test_meta_scalar_emit(self):
        rows = self.query("""
            SELECT gr_dynin.metadata_scalar_emit('abc', CAST(99 AS DOUBLE))
            FROM DUAL
        """)
        self.assertEqual('2', rows[0][0])

    def test_basic_set_emit_constants(self):
        rows = self.query("""
            SELECT gr_dynin.basic_set_emit(CAST(99 AS DOUBLE), '77', 'aaaa')
            FROM DUAL
        """)
        self.assertEqual(3, len(rows))
        self.assertIn(rows[0][0], ['99', '99.0'])
        self.assertEqual('77', rows[1][0])
        self.assertEqual('aaaa', rows[2][0])

    def test_basic_set_emit(self):
        rows = self.query("""
            SELECT gr_dynin.basic_set_emit(n, v)
            FROM gr_dynin_data.groupt
            GROUP BY id
            ORDER BY 1
        """)
        self.assertTrue(len(rows) >= 4)

    def test_basic_set_return_constants(self):
        rows = self.query("""
            SELECT gr_dynin.basic_set_return(CAST(99 AS DOUBLE), '77', 'aaaa')
            FROM DUAL
        """)
        self.assertEqual(1, len(rows))
        self.assertIn('99', rows[0][0])
        self.assertIn('77', rows[0][0])
        self.assertIn('aaaa', rows[0][0])

    def test_basic_set_return(self):
        rows = self.query("""
            SELECT gr_dynin.basic_set_return(n, v)
            FROM gr_dynin_data.groupt
            GROUP BY id
            ORDER BY 1
        """)
        self.assertEqual(2, len(rows))
        self.assertIn('aa', rows[0][0])
        self.assertIn('ba', rows[1][0])


if __name__ == "__main__":
    udf.main()
