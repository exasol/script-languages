#!/usr/bin/env python3

from exasol_python_test_framework import udf


class GetConnectionRTest(udf.TestCase):
    _BUG_CONN_USER = (
        'ialjksdhfalskdjhflaskdjfhalskdjhflaksjdhflaksdjfhalksjdfhlaksjdhflaksjdhfalskjdfhalskdjhflaksjdhflaksjdfhlaksjsadajksdhfaksjdfhalksdjfhalksdjfhalksjdfhqwiueryqw;er;lkjqwe;rdhflaksjdfhlaksdjfhaabcdefghijklmnopqrstuvwxyz'
    )
    _BUG_CONN_PASSWORD = (
        'abcdeoqsdfgsdjfglksjdfhglskjdfhglskdjfglskjdfghuietyewlrkjthertrewerlkjhqwelrkjhqwerlkjnwqerlkjhqwerkjlhqwerlkjhqwerlkhqwerkljhqwerlkjhqwerfghijklmnopqrstuvwxyz'
    )

    def setUp(self):
        self.query("DROP SCHEMA gr_conn CASCADE", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_fooconn", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_bug_connection", ignore_errors=True)
        self.query("CREATE SCHEMA gr_conn")
        self.query("CREATE CONNECTION gr_conn_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'")
        self.query(
            """
            CREATE CONNECTION gr_conn_bug_connection TO ''
            USER '%s' IDENTIFIED BY '%s'
            """ % (self._BUG_CONN_USER, self._BUG_CONN_PASSWORD)
        )

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_conn.print_connection(conn VARCHAR(1000))
            EMITS (type VARCHAR(200), addr VARCHAR(2000000), usr VARCHAR(2000000), pwd VARCHAR(2000000)) AS
            run <- function(ctx) {
                c <- exa$get_connection(ctx$conn)
                ctx$emit(c$type, c$address, c$user, c$password)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SCALAR SCRIPT gr_conn.print_connection_v2(conn VARCHAR(1000))
            EMITS (type VARCHAR(200), addr VARCHAR(2000000), usr VARCHAR(2000000), pwd VARCHAR(2000000)) AS
            run <- function(ctx) {
                c <- exa$get_connection(ctx$conn)
                ctx$emit(c$type, c$address, c$user, c$password)
            };
        """))

        self.query(udf.fixindent("""
            CREATE OR REPLACE R SET SCRIPT gr_conn.print_connection_set_emits(conn VARCHAR(1000))
            EMITS (dummy INTEGER) AS
            run <- function(ctx) {
                while (ctx$next_row()) {
                    c <- exa$get_connection(ctx$conn)
                }
                ctx$emit(1)
            };
        """))

    def tearDown(self):
        self.query("DROP CONNECTION gr_conn_fooconn", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_bug_connection", ignore_errors=True)

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

    def test_print_existing_connection_v2(self):
        rows = self.query("""
            SELECT gr_conn.print_connection_v2('GR_CONN_FOOCONN')
            FROM DUAL
        """)
        self.assertRowsEqual([('password', 'a', 'b', 'c')], rows)

    def test_get_connection(self):
        for _ in range(10):
            row = self.query("""
                WITH ten AS (VALUES 0,1,2,3,4,5,6,7,8,9 AS p(x))
                SELECT count(*)
                FROM (
                    SELECT gr_conn.print_connection_set_emits('GR_CONN_BUG_CONNECTION')
                    FROM (SELECT a.x FROM ten a, ten, ten, ten, ten) v
                    GROUP BY mod(v.rownum, 4019)
                )
            """)[0]
            self.assertEqual(4019, row[0])


if __name__ == "__main__":
    udf.main()
