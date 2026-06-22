#!/usr/bin/env python3

from exasol_python_test_framework import udf


class GetConnectionRTest(udf.TestCase):
    def setUp(self):
        self.query("DROP SCHEMA gr_conn CASCADE", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_fooconn", ignore_errors=True)
        self.query("CREATE SCHEMA gr_conn")
        self.query("CREATE CONNECTION gr_conn_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'")

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_conn.print_connection(conn VARCHAR(1000))
            EMITS (type VARCHAR(200), addr VARCHAR(2000000), usr VARCHAR(2000000), pwd VARCHAR(2000000)) AS
            run <- function(ctx) {
                c <- exa$get_connection(ctx$conn)
                ctx$emit(c$type, c$address, c$user, c$password)
            };
        """))

    def tearDown(self):
        self.query("DROP CONNECTION gr_conn_fooconn", ignore_errors=True)

    def test_print_existing_connection(self):
        rows = self.query("""
            SELECT gr_conn.print_connection('GR_CONN_FOOCONN')
            FROM DUAL
        """)
        self.assertRowsEqual([('password', 'a', 'b', 'c')], rows)

    def test_connection_not_found(self):
        with self.assertRaisesRegex(Exception, 'connection .* does not exist'):
            self.query("""
                SELECT gr_conn.print_connection('MISSING_CONN')
                FROM DUAL
            """)


if __name__ == "__main__":
    udf.main()
