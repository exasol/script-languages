#!/usr/bin/env python3

import locale

from exasol_python_test_framework import udf

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class UnicodeRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_unicode CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_unicode")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_unicode.unicode_len(word VARCHAR(1000))
            RETURNS INT AS
            run <- function(ctx) {
                if (is.null(ctx$word)) {
                    return(NULL)
                }
                nchar(ctx$word, type = 'chars')
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_unicode.unicode_upper(word VARCHAR(1000))
            RETURNS VARCHAR(1000) AS
            run <- function(ctx) {
                if (is.null(ctx$word)) {
                    return(NULL)
                }
                toupper(ctx$word)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_unicode.unicode_count(word VARCHAR(1000))
            EMITS (uchar VARCHAR(10), count INT) AS
            run <- function(ctx) {
                if (is.null(ctx$word)) {
                    return(NULL)
                }
                chars <- strsplit(ctx$word, '')[[1]]
                tab <- table(chars)
                for (n in names(tab)) {
                    ctx$emit(n, as.integer(tab[[n]]))
                }
            };
        """))

    def test_unicode_len(self):
        rows = self.query("""
            SELECT gr_unicode.unicode_len('ÄÖÜ')
            FROM DUAL
        """)
        self.assertRowsEqual([(3,)], rows)

    # R-only lightweight umlaut upper-case smoke test.
    def test_unicode_upper(self):
        rows = self.query("""
            SELECT gr_unicode.unicode_upper('äöü')
            FROM DUAL
        """)
        self.assertRowsEqual([('ÄÖÜ',)], rows)

    def test_unicode_count(self):
        rows = self.query("""
            SELECT gr_unicode.unicode_count('aab')
            FROM DUAL
            ORDER BY 1
        """)
        self.assertRowsEqual([('a', 2), ('b', 1)], rows)

    def _assert_unicode_char_roundtrip(self, codepoint):
        rows = self.query("""
            SELECT count, unicode(uchar) AS u
            FROM (
                SELECT gr_unicode.unicode_count(unicodechr(%d))
                FROM DUAL
            )
        """ % codepoint)
        self.assertRowsEqual([(1, codepoint)], rows)

    def test_unicode(self):
        for codepoint in (65, 255, 382, 65279, 63882, 65534, 66432, 173746, 1114111):
            self._assert_unicode_char_roundtrip(codepoint)

    def test_unicode_upper_is_subset_of_Unicode520_part2(self):
        rows = self.query("""
            SELECT codepoint
            FROM (
                SELECT 181 AS codepoint FROM DUAL
                UNION ALL
                SELECT 8126 AS codepoint FROM DUAL
            ) cp
            WHERE upper(unicodechr(codepoint)) != gr_unicode.unicode_upper(unicodechr(codepoint))
                AND unicodechr(codepoint) != gr_unicode.unicode_upper(unicodechr(codepoint))
            ORDER BY codepoint
        """)
        self.assertRowsEqual([], rows)

    def test_unicode_upper_is_subset_of_Unicode520_part3(self):
        rows = self.query("""
            SELECT codepoint
            FROM (
                SELECT 1010 AS codepoint FROM DUAL
            ) cp
            WHERE upper(unicodechr(codepoint)) != gr_unicode.unicode_upper(unicodechr(codepoint))
                AND unicodechr(codepoint) != gr_unicode.unicode_upper(unicodechr(codepoint))
            ORDER BY codepoint
        """)
        self.assertRowsEqual([], rows)


if __name__ == "__main__":
    udf.main()
