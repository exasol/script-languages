#!/usr/bin/env python3

from exasol_python_test_framework import udf


class _PathologicalFunctionsBase(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_path CASCADE", ignore_errors=True)
        self.query("CREATE SCHEMA gr_path")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_path.sleep(sec DOUBLE)
            RETURNS DOUBLE AS
            run <- function(ctx) {
                Sys.sleep(ctx$sec)
                ctx$sec
            };
        """))

class Test(_PathologicalFunctionsBase):
    def test_query_timeout(self):
        self.query("ALTER SESSION SET QUERY_TIMEOUT = 10")
        try:
            with self.assertRaisesRegex(Exception, 'Successfully reconnected after query timeout'):
                self.query("SELECT gr_path.sleep(100) FROM DUAL")
        finally:
            self.query("ALTER SESSION SET QUERY_TIMEOUT = 0")


if __name__ == "__main__":
    udf.main()
