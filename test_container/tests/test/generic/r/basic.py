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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_basic.basic_nth_partial_sum(n INTEGER)
            RETURNS INTEGER AS
            run <- function(ctx) {
                if (is.null(ctx$n)) return(0L)
                as.integer(ctx$n * (ctx$n + 1) / 2)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_basic.basic_sum_grp(x INTEGER)
            EMITS (s INTEGER) AS
            run <- function(ctx) {
                s <- 0L
                repeat {
                    if (!is.null(ctx$x)) {
                        s <- s + as.integer(ctx$x)
                    }
                    if (!ctx$next_row()) break
                }
                ctx$emit(s)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_basic.basic_emit_several_groups(a INTEGER, b INTEGER)
            EMITS (i INTEGER, j VARCHAR(40)) AS
            run <- function(ctx) {
                for (n in seq_len(as.integer(ctx$a))) {
                    for (i in 0:(as.integer(ctx$b) - 1)) {
                        ctx$emit(as.integer(i), as.character(exa$meta$vm_id))
                    }
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_basic.basic_test_reset(i INTEGER, j VARCHAR(40))
            EMITS (k INTEGER) AS
            run <- function(ctx) {
                ctx$emit(ctx$i)
                ctx$next_row()
                ctx$emit(ctx$i)
                ctx$reset()
                ctx$emit(ctx$i)
                ctx$next_row()
                ctx$emit(ctx$i)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_basic.performance_map_characters(text VARCHAR(1000))
            EMITS (w CHAR(1), c INTEGER) AS
            run <- function(ctx) {
                if (!is.null(ctx$text)) {
                    chars <- strsplit(ctx$text, '')[[1]]
                    for (ch in chars) {
                        ctx$emit(ch, 1L)
                    }
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_basic.performance_reduce_characters(w CHAR(1), c INTEGER)
            EMITS (w CHAR(1), c INTEGER) AS
            run <- function(ctx) {
                cnt <- 0L
                w <- ctx$w
                if (!is.null(w)) {
                    repeat {
                        cnt <- cnt + 1L
                        if (!ctx$next_row()) break
                    }
                    ctx$emit(w, cnt)
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

    def test_simple_combination(self):
        rows = self.query("""
            SELECT gr_basic.basic_sum(psum)
            FROM (
                SELECT gr_basic.basic_nth_partial_sum(n) AS psum
                FROM (
                    SELECT gr_basic.basic_range(10)
                    FROM DUAL
                )
            )
        """)
        self.assertRowsEqual([(165,)], rows)

    def test_simple_combination_grouping(self):
        rows = self.query("""
            SELECT gr_basic.basic_sum_grp(psum)
            FROM (
                SELECT MOD(N, 3) AS n,
                    gr_basic.basic_nth_partial_sum(n) AS psum
                FROM (
                    SELECT gr_basic.basic_range(10)
                    FROM DUAL
                )
            )
            GROUP BY n
            ORDER BY 1
        """)
        self.assertRowsEqual([(39,), (54,), (72,)], rows)

    def test_reset(self):
        rows = self.query("""
            SELECT gr_basic.basic_test_reset(i, j)
            FROM (SELECT gr_basic.basic_emit_several_groups(16, 8) FROM DUAL)
            GROUP BY i
            ORDER BY 1
        """)
        self.assertRowsEqual([(0,), (0,), (0,), (0,), (1,), (1,), (1,), (1,), (2,)], rows[:9])

    def test_order_by_clause(self):
        rows = self.query("""
            SELECT gr_basic.performance_reduce_characters(w, c)
            FROM (
               SELECT gr_basic.performance_map_characters('hello hello hello abc')
               FROM DUAL
            )
            GROUP BY w
            ORDER BY c DESC
        """)
        unsorted_list = [tuple(x) for x in rows]
        sorted_list = sorted(unsorted_list, key=lambda x: x[1], reverse=True)
        self.assertEqual(sorted_list, unsorted_list)

    def test_set_returns_has_empty_input_group_by(self):
        self.query("""
            SELECT gr_basic.set_returns_has_empty_input(c)
            FROM gr_basic_data.empty_table GROUP BY 'X'
        """)
        self.assertEqual(0, self.rowcount())

    def test_set_returns_has_empty_input_no_group_by(self):
        rows = self.query("""
            SELECT gr_basic.set_returns_has_empty_input(c)
            FROM gr_basic_data.empty_table
        """)
        self.assertRowsEqual([(None,)], rows)

    # R-only compatibility alias kept for historical naming in downstream runs.
    def test_set_with_empty_input(self):
        rows = self.query("""
            SELECT gr_basic.set_returns_has_empty_input(c)
            FROM gr_basic_data.empty_table
        """)
        self.assertRowsEqual([(None,)], rows)

    def test_set_emits_has_empty_input_group_by(self):
        self.query("""
            SELECT gr_basic.set_emits_has_empty_input(c)
            FROM gr_basic_data.empty_table GROUP BY 'X'
        """)
        self.assertEqual(0, self.rowcount())

    def test_set_emits_has_empty_input_no_group_by(self):
        rows = self.query("""
            SELECT gr_basic.set_emits_has_empty_input(c)
            FROM gr_basic_data.empty_table
        """)
        self.assertRowsEqual([(None, None)], rows)


if __name__ == "__main__":
    udf.main()
