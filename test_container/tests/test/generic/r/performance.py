#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import skip


class _PerformanceBase(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_perf CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_perf")
        self.query("""
            CREATE OR REPLACE TABLE gr_perf.wiki_names(text VARCHAR(350))
        """)
        self.query("""
            INSERT INTO gr_perf.wiki_names VALUES
                ('aba'),
                ('cab'),
                ('bbb'),
                (NULL)
        """)
        self.query("""
            CREATE OR REPLACE TABLE gr_perf.wiki_freq(w VARCHAR(1), c INTEGER)
        """)
        self.query("""
            INSERT INTO gr_perf.wiki_freq VALUES
                ('b', 5),
                ('a', 3),
                ('c', 1)
        """)

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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_counts_fast0(w VARCHAR(1000), c INTEGER)
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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_counts_fast7(w VARCHAR(1000), c INTEGER)
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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_counts_fast77(w VARCHAR(1000), c INTEGER)
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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_counts_fast777(w VARCHAR(1000), c INTEGER)
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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_counts_fast7777(w VARCHAR(1000), c INTEGER)
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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_counts_fast777777(w VARCHAR(1000), c INTEGER)
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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_counts_fast77777777(w VARCHAR(1000), c INTEGER)
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

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_perf.performance_map_characters(txt VARCHAR(2000))
            EMITS (w VARCHAR(1), c INTEGER) AS
            run <- function(ctx) {
                if (is.null(ctx$txt)) {
                    return(NULL)
                }
                chars <- unlist(strsplit(ctx$txt, ''))
                for (ch in chars) {
                    if (ch != '') {
                        ctx$emit(ch, 1L)
                    }
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_perf.performance_map_characters_fast(txt VARCHAR(2000))
            EMITS (w VARCHAR(1), c INTEGER) AS
            run <- function(ctx) {
                if (is.null(ctx$txt)) {
                    return(NULL)
                }
                chars <- unlist(strsplit(ctx$txt, ''))
                for (ch in chars) {
                    if (ch != '') {
                        ctx$emit(ch, 1L)
                    }
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_perf.performance_map_characters_fast0(txt VARCHAR(2000))
            EMITS (w VARCHAR(1), c INTEGER) AS
            run <- function(ctx) {
                if (is.null(ctx$txt)) {
                    return(NULL)
                }
                chars <- unlist(strsplit(ctx$txt, ''))
                for (ch in chars) {
                    if (ch != '') {
                        ctx$emit(ch, 1L)
                    }
                }
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_characters(w VARCHAR(1), c INTEGER)
            EMITS (w VARCHAR(1), c INTEGER) AS
            run <- function(ctx) {
                ch <- ctx$w
                cnt <- as.integer(ctx$c)
                while (ctx$next_row()) {
                    cnt <- cnt + as.integer(ctx$c)
                }
                ctx$emit(ch, cnt)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_perf.performance_reduce_characters_fast(w VARCHAR(1), c INTEGER)
            EMITS (w VARCHAR(1), c INTEGER) AS
            run <- function(ctx) {
                ch <- ctx$w
                cnt <- as.integer(ctx$c)
                while (ctx$next_row()) {
                    cnt <- cnt + as.integer(ctx$c)
                }
                ctx$emit(ch, cnt)
            };
        """))

    def _assert_word_count_with_reducer(self, reducer_name):
        rows = self.query("""
            SELECT COUNT(*)
            FROM (
                SELECT %s(w, c)
                FROM (
                    SELECT gr_perf.performance_map_words(varchar02)
                    FROM (
                        VALUES
                            ('hello hello exasol'),
                            ('exasol rocks')
                        AS t(varchar02)
                    )
                )
                GROUP BY w
            )
        """ % reducer_name)
        self.assertRowsEqual([(3,)], rows)

class PerformanceROnly(_PerformanceBase):
    """R-only tests without a generic counterpart."""

    # R-only small deterministic smoke query complementing heavy word-count tests.
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

class WordCount(_PerformanceBase):
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

    def test_word_count(self):
        self._assert_word_count_with_reducer('gr_perf.performance_reduce_counts')

    def test_word_count_fast0(self):
        self._assert_word_count_with_reducer('gr_perf.performance_reduce_counts_fast0')

    def test_word_count_fast7(self):
        self._assert_word_count_with_reducer('gr_perf.performance_reduce_counts_fast7')

    def test_word_count_fast77(self):
        self._assert_word_count_with_reducer('gr_perf.performance_reduce_counts_fast77')

    def test_word_count_fast777(self):
        self._assert_word_count_with_reducer('gr_perf.performance_reduce_counts_fast777')

    def test_word_count_fast7777(self):
        self._assert_word_count_with_reducer('gr_perf.performance_reduce_counts_fast7777')

    def test_word_count_fast777777(self):
        self._assert_word_count_with_reducer('gr_perf.performance_reduce_counts_fast777777')

    def test_word_count_fast77777777(self):
        self._assert_word_count_with_reducer('gr_perf.performance_reduce_counts_fast77777777')


class FrequencyAnalysis(_PerformanceBase):
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

    @skip('csv data for tables wiki_freq and wiki_names is currently not available')
    def test_frequency_analysis(self):
        pass

    @skip('csv data for tables wiki_freq and wiki_names is currently not available')
    def test_frequency_analysis_fast(self):
        pass

    @skip('csv data for tables wiki_freq and wiki_names is currently not available')
    def test_frequency_analysis_fast0(self):
        pass


if __name__ == "__main__":
    udf.main()
