#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest


class GetConnectionRTest(udf.TestCase):
    _BUG_CONN_USER = (
        'ialjksdhfalskdjhflaskdjfhalskdjhflaksjdhflaksdjfhalksjdfhlaksjdhflaksjdhfalskjdfhalskdjhflaksjdhflaksjdfhlaksjsadajksdhfaksjdfhalksdjfhalksdjfhalksjdfhqwiueryqw;er;lkjqwe;rdhflaksjdfhlaksdjfhaabcdefghijklmnopqrstuvwxyz'
    )
    _BUG_CONN_PASSWORD = (
        'abcdeoqsdfgsdjfglksjdfhglskjdfhglskdjfglskjdfghuietyewlrkjthertrewerlkjhqwelrkjhqwerlkjnwqerlkjhqwerkjlhqwerlkjhqwerlkhqwerkljhqwerlkjhqwerfghijklmnopqrstuvwxyz'
    )
    _BIG_CONN_ADDRESS = "a" * 2 * 1000 * 100
    _BIG_CONN_USER = "u" * 2 * 1000 * 100
    _BIG_CONN_PASSWORD = "p" * 2 * 1000 * 100

    def _drop_common_objects(self):
        self.query("DROP FORCE VIRTUAL SCHEMA VS CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA adapter CASCADE", ignore_errors=True)
        self.query("DROP SCHEMA spot42542 CASCADE", ignore_errors=True)
        self.query("DROP USER foo CASCADE", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_fooconn", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_bug_connection", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_ac_fooconn", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_largest_conn", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_my_conn1", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_my_conn2", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_my_conn3", ignore_errors=True)
        self.query("DROP CONNECTION gr_conn_my_conn", ignore_errors=True)

    def _create_user(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username=username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username=username, password=password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

    def _get_client(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user %s', username)
        client.connect(uid=username, pwd=password)
        return client

    def setUp(self):
        self.query("DROP SCHEMA gr_conn CASCADE", ignore_errors=True)
        self._drop_common_objects()
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
        self._drop_common_objects()

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

    def test_sys_priv_exists(self):
        sys_priv = self.query("SELECT * FROM EXA_DBA_SYS_PRIVS WHERE PRIVILEGE = 'ACCESS ANY CONNECTION'")
        self.assertRowsEqual([("DBA", "ACCESS ANY CONNECTION", True)], sys_priv)

    def test_use_connection_without_rights(self):
        self.query("CREATE CONNECTION gr_conn_ac_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'", ignore_errors=True)
        self._create_user('foo', 'foo')
        self.query('GRANT CREATE SCHEMA TO foo')
        self.query('GRANT CREATE SCRIPT TO foo')
        self.query('GRANT EXECUTE ON SCRIPT gr_conn.print_connection TO foo')
        self.commit()
        foo_conn = self._get_client('foo', 'foo')
        with self.assertRaisesRegex(Exception, 'insufficient privileges for using connection GR_CONN_AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query("""
                SELECT gr_conn.print_connection('GR_CONN_AC_FOOCONN')
            """)
        foo_conn.commit()

    def test_use_connection_with_old_right(self):
        self.query("CREATE CONNECTION gr_conn_ac_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'", ignore_errors=True)
        self._create_user('foo', 'foo')
        self.query('GRANT CREATE SCHEMA TO foo')
        self.query('GRANT CREATE SCRIPT TO foo')
        self.query('GRANT CONNECTION gr_conn_ac_fooconn TO foo')
        self.query('GRANT EXECUTE ON SCRIPT gr_conn.print_connection TO foo')
        self.commit()
        foo_conn = self._get_client('foo', 'foo')
        with self.assertRaisesRegex(Exception, 'insufficient privileges for using connection GR_CONN_AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query("""
                SELECT gr_conn.print_connection('GR_CONN_AC_FOOCONN')
            """)
        foo_conn.commit()

    def test_use_connection_with_new_right(self):
        self.query("CREATE CONNECTION gr_conn_ac_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'", ignore_errors=True)
        self._create_user('foo', 'foo')
        self.query('GRANT CREATE SCHEMA TO foo')
        self.query('GRANT CREATE SCRIPT TO foo')
        self.query('GRANT EXECUTE ON SCRIPT gr_conn.print_connection TO foo')
        self.query('GRANT ACCESS ON CONNECTION gr_conn_ac_fooconn TO foo')
        self.commit()
        foo_conn = self._get_client('foo', 'foo')
        rows = foo_conn.query("""
            SELECT gr_conn.print_connection('GR_CONN_AC_FOOCONN')
        """)
        self.assertRowsEqual([('password', 'a', 'b', 'c')], rows)
        foo_conn.commit()

    def test_use_connection_in_old_import_with_new_right(self):
        self.query("CREATE CONNECTION gr_conn_ac_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'", ignore_errors=True)
        self._create_user('foo', 'foo')
        self.query('GRANT INSERT ANY TABLE TO foo')
        self.query('GRANT IMPORT TO foo', ignore_errors=True)
        self.query('GRANT ACCESS ON CONNECTION gr_conn_ac_fooconn TO foo')
        self.commit()
        foo_conn = self._get_client('foo', 'foo')
        with self.assertRaisesRegex(Exception, 'insufficient privileges for using connection'):
            foo_conn.query("""
                IMPORT FROM FBV AT GR_CONN_AC_FOOCONN FILE 'foo'
            """)
        foo_conn.commit()

    def test_get_big_connection(self):
        self.query(
            """
            CREATE CONNECTION gr_conn_largest_conn TO '{address}' USER '{user}' IDENTIFIED BY '{password}'
            """.format(
                address=self._BIG_CONN_ADDRESS,
                user=self._BIG_CONN_USER,
                password=self._BIG_CONN_PASSWORD,
            )
        )
        rows = self.query("""
            SELECT gr_conn.print_connection('GR_CONN_LARGEST_CONN')
        """)
        self.assertRowsEqual([
            ('password', self._BIG_CONN_ADDRESS, self._BIG_CONN_USER, self._BIG_CONN_PASSWORD)
        ], rows)

    def test_access_connection_in_adapter(self):
        self.query("CREATE SCHEMA IF NOT EXISTS adapter")
        self.query("CREATE CONNECTION gr_conn_my_conn TO 'MYADDRESS' USER 'MYUSER' IDENTIFIED BY 'MYPASSWORD'")
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 ADAPTER SCRIPT adapter.fast_adapter_conn AS
            import json

            def adapter_call(request):
                root = json.loads(request)
                if root["type"] == "createVirtualSchema":
                    c = exa.get_connection("GR_CONN_MY_CONN")
                    res = {
                        "type": "createVirtualSchema",
                        "schemaMetadata": {
                            "tables": [{
                                "name": "T1",
                                "columns": [{
                                    "name": c.address,
                                    "dataType": {"type": "VARCHAR", "size": 2000000}
                                }, {
                                    "name": c.user,
                                    "dataType": {"type": "VARCHAR", "size": 2000000}
                                }, {
                                    "name": c.password,
                                    "dataType": {"type": "VARCHAR", "size": 2000000}
                                }]
                            }]
                        }
                    }
                    return json.dumps(res)
                if root["type"] == "dropVirtualSchema":
                    return json.dumps({"type": "dropVirtualSchema"})
                raise ValueError('Unsupported callback')
            /
        '''))
        self.query("CREATE VIRTUAL SCHEMA VS USING adapter.fast_adapter_conn")
        rows = self.query("SELECT COLUMN_NAME FROM EXA_ALL_COLUMNS WHERE COLUMN_TABLE='T1' ORDER BY COLUMN_NAME")
        self.assertRowsEqual([('MYADDRESS',), ('MYPASSWORD',), ('MYUSER',)], rows)

    def test_no_user_and_no_identified_by(self):
        self.query("CREATE OR REPLACE CONNECTION gr_conn_my_conn1 TO 'MYADDRESS'")
        rows = self.query("SELECT gr_conn.print_connection('GR_CONN_MY_CONN1')")
        self.assertRowsEqual([('password', 'MYADDRESS', None, None)], rows)

    def test_no_user(self):
        self.query("CREATE OR REPLACE CONNECTION gr_conn_my_conn2 TO 'MYADDRESS' IDENTIFIED BY 'MYPASSWORD'")
        rows = self.query("SELECT gr_conn.print_connection('GR_CONN_MY_CONN2')")
        self.assertRowsEqual([('password', 'MYADDRESS', None, 'MYPASSWORD')], rows)

    def test_no_identified_by(self):
        self.query("CREATE OR REPLACE CONNECTION gr_conn_my_conn3 TO 'MYADDRESS' USER 'MYUSER'")
        rows = self.query("SELECT gr_conn.print_connection('GR_CONN_MY_CONN3')")
        self.assertRowsEqual([('password', 'MYADDRESS', 'MYUSER', None)], rows)

    def test_use_connection_udfs_in_view(self):
        self.query("CREATE CONNECTION gr_conn_ac_fooconn TO 'a' USER 'b' IDENTIFIED BY 'c'", ignore_errors=True)
        self._create_user('foo', 'foo')
        self.query('CREATE SCHEMA IF NOT EXISTS spot42542')
        self.query(
            "CREATE OR REPLACE VIEW spot42542.print_connection_wrapper AS "
            "SELECT gr_conn.print_connection('GR_CONN_AC_FOOCONN')"
        )
        self.query('GRANT SELECT ON spot42542.print_connection_wrapper TO foo')
        self.commit()
        foo_conn = self._get_client('foo', 'foo')
        rows = foo_conn.query('SELECT * FROM spot42542.print_connection_wrapper')
        foo_conn.commit()
        self.assertRowsEqual([('password', 'a', 'b', 'c')], rows)


if __name__ == "__main__":
    udf.main()
