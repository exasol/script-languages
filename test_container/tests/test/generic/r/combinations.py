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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_combi.basic_range(n INTEGER)
            EMITS (n INTEGER) AS
            run <- function(ctx) {
                if (!is.null(ctx$n)) {
                    for (i in 0:(ctx$n - 1)) {
                        ctx$emit(as.integer(i))
                    }
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_combi.basic_sum(x INTEGER)
            RETURNS INTEGER AS
            run <- function(ctx) {
                s <- as.integer(0)
                if (!is.null(ctx$x)) s <- s + as.integer(ctx$x)
                while (ctx$next_row()) {
                    if (!is.null(ctx$x)) s <- s + as.integer(ctx$x)
                }
                s
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_combi.basic_sum_grp(x INTEGER)
            EMITS (s INTEGER) AS
            run <- function(ctx) {
                s <- 0L
                repeat {
                    if (!is.null(ctx$x)) s <- s + as.integer(ctx$x)
                    if (!ctx$next_row()) break
                }
                ctx$emit(s)
            };
        """))

    def test_scalar_returns(self):
        rows = self.query("""
            SELECT ROUND(gr_combi.scalar_returns(x, y) / 2)
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(0,), (0,)], rows)

    def test_scalar_emits(self):
        rows = self.query("""
            SELECT gr_combi.scalar_emits(x * 10, y * 10)
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(1.0, 1.0), (2.0, 4.0)], rows)

    def test_two_scalar_returns(self):
        rows = self.query("""
            SELECT
                gr_combi.scalar_returns(
                    gr_combi.scalar_returns(x * 10, y * 10),
                    gr_combi.scalar_returns(y * 10, x * 10)
                )
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(6.0,), (6.0,)], rows)

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

    # --- 2-ary: scalar_returns outer ---

    def test_scalar_returns_scalar_emits(self):
        rows = self.query("""
            SELECT gr_combi.scalar_returns(x * 10, y * 10)
            FROM (
                SELECT gr_combi.scalar_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(20.0,), (60.0,)], rows)

    def test_scalar_returns_set_returns_inline(self):
        rows = self.query("""
            SELECT
                gr_combi.scalar_returns(
                    gr_combi.set_returns(x * 10, y * 10),
                    gr_combi.set_returns(x * 10, y * 10)
                )
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(12.0,)], rows)

    def test_scalar_returns_set_returns_1(self):
        rows = self.query("""
            SELECT gr_combi.scalar_returns(a, 5)
            FROM (
                SELECT gr_combi.set_returns(x * 10, y * 10) AS a
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(11.0,)], rows)

    def test_scalar_returns_set_returns_2(self):
        rows = self.query("""
            SELECT gr_combi.scalar_returns(aa.a, bb.a)
            FROM (
                SELECT gr_combi.set_returns(x * 10, y * 20) AS a
                FROM gr_combi_data.small
            ) AS aa,
            (
                SELECT gr_combi.set_returns(x * 20, y * 10) AS a
                FROM gr_combi_data.small
            ) AS bb
        """)
        self.assertRowsEqual([(18.0,)], rows)

    def test_scalar_returns_set_emits_1(self):
        rows = self.query("""
            SELECT gr_combi.scalar_returns(x * 10, y * 10)
            FROM (
                SELECT gr_combi.set_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(30.0,), (30.0,)], rows)

    def test_scalar_returns_set_emits_2(self):
        rows = self.query("""
            SELECT gr_combi.scalar_returns(aa.x * 10, bb.x * 10)
            FROM (
                SELECT gr_combi.set_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            ) AS aa,
            (
                SELECT gr_combi.set_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            ) AS bb
            WHERE aa.x = bb.y AND aa.y = bb.x
        """)
        self.assertRowsEqual([(30.0,), (30.0,)], rows)

    # --- 2-ary: scalar_emits outer ---

    def test_scalar_emits_scalar_returns_inline(self):
        rows = self.query("""
            SELECT
                gr_combi.scalar_emits(
                    gr_combi.scalar_returns(x * 10, y * 10),
                    gr_combi.scalar_returns(y * 10, x * 10)
                )
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(3.0, 9.0), (3.0, 9.0)], rows)

    def test_scalar_emits_scalar_returns(self):
        rows = self.query("""
            SELECT gr_combi.scalar_emits(a, b)
            FROM (
                SELECT
                    gr_combi.scalar_returns(x * 10, y * 10) AS a,
                    gr_combi.scalar_returns(y * 10, x * 10) AS b
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(3.0, 9.0), (3.0, 9.0)], rows)

    def test_scalar_emits_set_returns(self):
        rows = self.query("""
            SELECT gr_combi.scalar_emits(a, b)
            FROM (
                SELECT gr_combi.set_returns(x * 10, y * 10) AS a
                FROM gr_combi_data.small
            ),
            (
                SELECT gr_combi.set_returns(x * 10, y * 10) AS b
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(6.0, 36.0)], rows)

    def test_scalar_emits_set_emits(self):
        rows = self.query("""
            SELECT gr_combi.scalar_emits(x * 10, y * 10)
            FROM (
                SELECT gr_combi.set_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            )
        """)
        r = [(float(i), float(i * i)) for i in range(10, 21)]
        self.assertRowsEqual(r, rows)

    # --- 2-ary: set_returns outer ---

    def test_set_returns_scalar_returns(self):
        rows = self.query("""
            SELECT
                gr_combi.set_returns(
                    gr_combi.scalar_returns(x * 10, y * 10),
                    gr_combi.scalar_returns(y * 10, x * 10)
                )
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(12.0,)], rows)

    def test_set_returns_scalar_emits(self):
        rows = self.query("""
            SELECT gr_combi.set_returns(x * 10, y * 10)
            FROM (
                SELECT gr_combi.scalar_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(80.0,)], rows)

    def test_set_returns_set_returns_inline(self):
        with self.assertRaisesRegex(Exception, 'encapsulated set function'):
            self.query("""
                SELECT
                    gr_combi.set_returns(
                        gr_combi.set_returns(x * 10, y * 10),
                        gr_combi.set_returns(x * 10, y * 10)
                    )
                FROM gr_combi_data.small
            """)

    def test_set_returns_set_returns(self):
        rows = self.query("""
            SELECT gr_combi.set_returns(a, b)
            FROM (
                SELECT gr_combi.set_returns(x * 20, y * 30) AS a
                FROM gr_combi_data.small
            ),
            (
                SELECT gr_combi.set_returns(x * 50, y * 70) AS b
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(51.0,)], rows)

    def test_set_returns_set_emits(self):
        rows = self.query("""
            SELECT gr_combi.set_returns(x * 10, y * 10)
            FROM (
                SELECT gr_combi.set_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(60.0,)], rows)

    # --- 2-ary: set_emits outer ---

    def test_set_emits_scalar_returns(self):
        rows = self.query("""
            SELECT
                gr_combi.set_emits(
                    gr_combi.scalar_returns(x * 10, y * 10),
                    gr_combi.scalar_returns(y * 10, x * 10)
                )
            FROM gr_combi_data.small
        """)
        self.assertRowsEqual([(3.0, 3.0), (3.0, 3.0)], rows)

    def test_set_emits_scalar_emits(self):
        rows = self.query("""
            SELECT gr_combi.set_emits(x * 10, y * 10)
            FROM (
                SELECT gr_combi.scalar_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            )
            ORDER BY x, y
        """)
        self.assertRowsEqual([(10.0, 10.0), (40.0, 20.0)], rows)

    def test_set_emits_set_returns_inline(self):
        with self.assertRaisesRegex(Exception, 'encapsulated set function'):
            self.query("""
                SELECT
                    gr_combi.set_emits(
                        gr_combi.set_returns(x * 10, y * 10),
                        gr_combi.set_returns(x * 10, y * 10)
                    )
                FROM gr_combi_data.small
            """)

    def test_set_emits_set_returns(self):
        rows = self.query("""
            SELECT gr_combi.set_emits(a, b)
            FROM (
                SELECT gr_combi.set_returns(x * 20, y * 30) AS a
                FROM gr_combi_data.small
            ),
            (
                SELECT gr_combi.set_returns(x * 50, y * 70) AS b
                FROM gr_combi_data.small
            )
        """)
        self.assertRowsEqual([(36.0, 15.0)], rows)

    def test_set_emits_set_emits(self):
        rows = self.query("""
            SELECT gr_combi.set_emits(x * 10, y * 10)
            FROM (
                SELECT gr_combi.set_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            )
            ORDER BY x, y
        """)
        self.assertRowsEqual([(10.0, 20.0), (20.0, 10.0)], rows)

    # --- 3-ary ---

    def test_3ary_set_returns_set_emits_scalar_emits(self):
        rows = self.query("""
            SELECT gr_combi.basic_sum(s)
            FROM (
                SELECT gr_combi.basic_sum_grp(n)
                FROM (
                    SELECT gr_combi.basic_range(10)
                    FROM DUAL
                )
            )
        """)
        self.assertRowsEqual([(45,)], rows)

    def test_3ary_set_returns_scalar_emits_scalar_emits(self):
        rows = self.query("""
            SELECT gr_combi.basic_sum(x)
            FROM (
                SELECT gr_combi.scalar_emits(n, n + 2)
                FROM (
                    SELECT gr_combi.basic_range(10)
                    FROM DUAL
                )
            )
        """)
        self.assertRowsEqual([(165,)], rows)

    def test_3ary_set_returns_scalar_returns_scalar_emits(self):
        rows = self.query("""
            SELECT gr_combi.basic_sum(x)
            FROM (
                SELECT gr_combi.scalar_returns(n, 2) AS x
                FROM (
                    SELECT gr_combi.basic_range(10)
                    FROM DUAL
                )
            )
        """)
        self.assertRowsEqual([(65,)], rows)

    def test_scalar_emits_scalar_emits(self):
        rows = self.query("""
            SELECT gr_combi.scalar_emits(x * 10, y * 10)
            FROM (
                SELECT gr_combi.scalar_emits(x * 10, y * 10)
                FROM gr_combi_data.small
            )
            ORDER BY x, y
        """)
        r = [(10.0, 100.0)]
        r.extend([(float(i), float(i * i)) for i in range(20, 41)])
        self.assertRowsEqual(r, rows)

    def test_scalar_emits_set_returns_inline(self):
        with self.assertRaisesRegex(Exception, 'encapsulated set function'):
            self.query("""
                SELECT
                    gr_combi.scalar_emits(
                        gr_combi.set_returns(x * 10, y * 10),
                        gr_combi.set_returns(x * 10, y * 10)
                    )
                FROM gr_combi_data.small
            """)

    def test_set_returns_set_emits_scalar_emits(self):
        self.test_3ary_set_returns_set_emits_scalar_emits()

    def test_set_returns_scalar_emits_scalar_emits(self):
        self.test_3ary_set_returns_scalar_emits_scalar_emits()

    def test_set_returns_scalar_returns_scalar_emits(self):
        self.test_3ary_set_returns_scalar_returns_scalar_emits()

    @staticmethod
    def _partial_sum(n, degree):
        def basic_range(k, d):
            if d == 0:
                return list(range(k))
            return sum([list(range(x)) for x in basic_range(k + 1, d - 1)], [])

        return len(basic_range(n, degree))

    def test_n_scalar_emits(self):
        for n in range(10):
            self.query(
                'SELECT gr_combi.basic_range(n+1) FROM (\n' * n +
                'SELECT gr_combi.basic_range(5) FROM DUAL\n' +
                ')' * n
            )
            self.assertEqual(self._partial_sum(5, n), self.rowcount())

    def test_set_returns_n_scalar_emits(self):
        for n in range(10):
            rows = self.query(
                'SELECT max(n) FROM (' +
                'SELECT gr_combi.basic_range(n+1) FROM (\n' * n +
                'SELECT gr_combi.basic_range(5) FROM DUAL\n' +
                ')' * (n + 1)
            )
            self.assertEqual(4, rows[0][0])


if __name__ == "__main__":
    udf.main()
