#!/usr/bin/env python3

from exasol_python_test_framework import udf


class PerformanceRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_perf CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_perf")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_perf.performance_map_words(w VARCHAR(1000))
            EMITS (w VARCHAR(1000), c INTEGER) AS
            run <- function(ctx) {
                if (is.null(ctx$w)) {
                    return(NULL)
                }
                parts <- unlist(strsplit(ctx$w, '[^[:alnum:]_]+'))
                parts <- parts[parts != '']
                for (p in parts) {
                    ctx$emit(p, 1L)
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_counts(w VARCHAR(1000), c INTEGER)
            EMITS (w VARCHAR(1000), c INTEGER) AS
            run <- function(ctx) {
                word <- ctx$w
                cnt <- as.integer(ctx$c)
                while (ctx$next_row()) {
                    cnt <- cnt + as.integer(ctx$c)
                }
                ctx$emit(word, cnt)
            };
        """))

    def test_word_count_query(self):
        rows = self.query("""
            SELECT COUNT(*)
            FROM (
                SELECT gr_perf.performance_reduce_counts(w, c)
                FROM (
                    SELECT gr_perf.performance_map_words('hello hello exasol')
                    FROM DUAL
                )
                GROUP BY w
            )
        """)
        self.assertRowsEqual([(2,)], rows)

    def test_word_unicode_count(self):
        rows = self.query("""
            SELECT COUNT(*)
            FROM (
                SELECT gr_perf.performance_reduce_counts(w, c)
                FROM (
                    SELECT gr_perf.performance_map_words('café latte café')
                    FROM DUAL
                )
                GROUP BY w
            )
        """)
        self.assertRowsEqual([(2,)], rows)

    def test_frequency_analysis_light(self):
        rows = self.query("""
            SELECT gr_perf.performance_reduce_counts(w, c)
            FROM (
                SELECT gr_perf.performance_map_words('the quick brown fox the fox')
                FROM DUAL
            )
            GROUP BY w
            ORDER BY c DESC
        """)
        words = dict(rows)
        self.assertEqual(2, words.get('the'))
        self.assertEqual(2, words.get('fox'))
        self.assertEqual(1, words.get('quick'))
        self.assertEqual(1, words.get('brown'))


if __name__ == "__main__":
    udf.main()
