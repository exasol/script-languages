#!/usr/bin/env python3

from exasol_python_test_framework import udf


class GenericTypesRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_types CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_types")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_boolean(x BOOLEAN)
            RETURNS BOOLEAN AS
            run <- function(ctx) {
                ctx$x
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_integer(x INTEGER)
            RETURNS INTEGER AS
            run <- function(ctx) {
                if (is.null(ctx$x)) {
                    return(NULL)
                }
                as.integer(ctx$x)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_double(x DOUBLE)
            RETURNS DOUBLE AS
            run <- function(ctx) {
                if (is.null(ctx$x)) {
                    return(NULL)
                }
                as.double(ctx$x)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_types.echo_varchar10(x VARCHAR(10))
            RETURNS VARCHAR(10) AS
            run <- function(ctx) {
                ctx$x
            };
        """))

    def test_echo_boolean(self):
        rows = self.query("""
            SELECT
                gr_types.echo_boolean(NULL) IS NULL,
                gr_types.echo_boolean(TRUE) = TRUE,
                gr_types.echo_boolean(FALSE) = FALSE
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True)], rows)

    def test_echo_integer(self):
        rows = self.query("""
            SELECT
                gr_types.echo_integer(NULL) IS NULL,
                gr_types.echo_integer(1) = 1,
                gr_types.echo_integer(-1) = -1
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True, True)], rows)

    def test_echo_double_and_varchar(self):
        rows = self.query("""
            SELECT
                gr_types.echo_double(1.5) = 1.5,
                gr_types.echo_varchar10('abc') = 'abc'
            FROM DUAL
        """)
        self.assertRowsEqual([(True, True)], rows)


if __name__ == "__main__":
    udf.main()
