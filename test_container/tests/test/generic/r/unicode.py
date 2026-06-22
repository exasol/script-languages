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


if __name__ == "__main__":
    udf.main()
