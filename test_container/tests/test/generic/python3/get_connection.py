#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest


class _Python3UdfSetup(udf.TestCase):
    def _setup_common_udfs(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Create Python3 UDFs for get_connection testing
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT PRINT_CONNECTION(conn varchar(1000))
            EMITS(type varchar(200), addr varchar(2000000), usr varchar(2000000), pwd varchar(2000000))
            AS
            def run(ctx):
                c = exa.get_connection(ctx.conn)
                ctx.emit(c.type, c.address, c.user, c.password)
            /
        '''))
        
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT PRINT_CONNECTION_SET_EMITS(conn varchar(1000))
            EMITS(type varchar(200), addr varchar(2000000), usr varchar(2000000), pwd varchar(2000000))
            AS
            def run(ctx):
                c = exa.get_connection(ctx.conn)
                ctx.emit(c.type, c.address, c.user, c.password)
            /
        '''))

    def setUp(self):
        self._setup_common_udfs()


class GetConnectionMemoryBug(_Python3UdfSetup):
    def setUp(self):
        self._setup_common_udfs()
        
        self.query(
            '''CREATE OR REPLACE CONNECTION test_get_connection_bug_connection TO '' USER 'ialjksdhfalskdjhflaskdjfhalskdjhflaksjdhflaksdjfhalksjdfhlaksjdhflaksjdhfalskjdfhalskdjhflaksjdhflaksjdfhlaksjsadajksdhfaksjdfhalksdjfhalksdjfhalksjdfhqwiueryqw;er;lkjqwe;rdhflaksjdfhlaksdjfhaabcdefghijklmnopqrstuvwxyz' IDENTIFIED BY 'abcdeoqsdfgsdjfglksjdfhglskjdfhglskdjfglskjdfghuietyewlrkjthertrewerlkjhqwelrkjhqwerlkjnwqerlkjhqwerkjlhqwerlkjhqwerlkhqwerkljhqwerlkjhqwerfghijklmnopqrstuvwxyz';''')

    def test_get_connection(self):
        for x in range(10):
            row = self.query(
                '''with ten as (values 0,1,2,3,4,5,6,7,8,9 as p(x)) select count(*) from (select fn1.print_connection_set_emits('test_get_connection_bug_connection') from (select a.x from ten a, ten, ten, ten, ten) v group by mod(v.rownum,4019))''')[
                0]
            self.assertEqual(4019, row[0])


class AccessConnectionSysPriv(udf.TestCase):
    def testSysPrivExists(self):
        sys_priv = self.query("SELECT * FROM EXA_DBA_SYS_PRIVS WHERE PRIVILEGE = 'ACCESS ANY CONNECTION'")
        self.assertRowsEqual([("DBA", "ACCESS ANY CONNECTION", True)], sys_priv)


class GetConnectionTest(_Python3UdfSetup):
    def setUp(self):
        self._setup_common_udfs()
        
        self.query('''
        create connection FOOCONN to 'a' user 'b' identified by 'c'
        ''')

    def tearDown(self):
        self.query('drop connection FOOCONN')

    def test_print_existing_connection(self):
        rows = self.query('''
            SELECT fn1.print_connection('FOOCONN')
            ''')
        self.assertRowsEqual([('password', 'a', 'b', 'c')], rows)

    def test_connection_not_found(self):
        with self.assertRaisesRegex(Exception, 'connection FOO does not exist'):
            self.query('''
                SELECT fn1.print_connection('FOO')
                ''')


class GetConnectionAccessControlTest(_Python3UdfSetup):
    def setUp(self):
        self._setup_common_udfs()
        
        self.query('''
        create connection AC_FOOCONN to 'a' user 'b' identified by 'c'
        ''', ignore_errors=True)

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid=username, pwd=password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username=username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username=username, password=password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

    def testUseConnectionWithoutRights(self):
        self.createUser("foo", "foo")
        self.query('grant create schema to foo')
        self.query('grant create script to foo')
        self.query('grant execute on script fn1.print_connection to foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        with self.assertRaisesRegex(Exception,
                                    'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                SELECT fn1.print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        self.query('drop user foo cascade')
        self.commit()

    def testUseConnectionWithOldRight(self):
        self.createUser("foo", "foo")
        self.query('grant create schema to foo')
        self.query('grant create script to foo')
        self.query('grant connection ac_fooconn to foo')
        self.query('grant execute on script fn1.print_connection to foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        with self.assertRaisesRegex(Exception,
                                    'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                 select fn1.print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        self.query('drop user foo cascade')
        self.commit()

    def testUseConnectionWithNewRight(self):
        self.createUser("foo", "foo")
        self.query('grant create schema to foo')
        self.query('grant create script to foo')
        self.query('grant execute on script fn1.print_connection to foo')
        self.query('GRANT ACCESS ON CONNECTION ac_fooconn to foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        rows = foo_conn.query('''
             select fn1.print_connection('AC_FOOCONN')
        ''')
        self.assertRowsEqual([('password', 'a', 'b', 'c')], rows)
        foo_conn.commit()
        self.query('drop user foo cascade')
        self.commit()

    def testUseConnectionInOldImportWithNewRight(self):
        self.createUser("foo", "foo")
        self.query('grant insert any table to foo')
        self.query('grant import to foo', ignore_errors=True)  # only supported and needed since 6.1
        self.query('GRANT ACCESS ON CONNECTION ac_fooconn to foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        with self.assertRaisesRegex(Exception, 'insufficient privileges for using connection'):
            foo_conn.query('''
                import from fbv at ac_fooconn file 'foo'
            ''')
        foo_conn.commit()
        self.query('drop user foo cascade')
        self.commit()


class BigConnectionTest(_Python3UdfSetup):
    # Should be max. size 2.000.000, but this will cause our odbc driver to crash (sigsegv) during logging (DWA-20290).
    # Will be increased to max size when bug is fixed
    address = "a" * 2 * 1000 * 100
    user = "u" * 2 * 1000 * 100
    password = "p" * 2 * 1000 * 100

    def setUp(self):
        self._setup_common_udfs()
        
        self.query('''
            create connection LARGEST_CONN to '{address}' user '{user}' identified by '{password}'
            '''.format(address=self.address, user=self.user, password=self.password))

    def tearDown(self):
        self.query("DROP CONNECTION LARGEST_CONN")

    def testGetBigConnection(self):
        rows = self.query('''
            SELECT fn1.print_connection('LARGEST_CONN')
            ''')
        self.assertRowsEqual([('password', self.address, self.user, self.password)], rows)


class ConnectionTest(udf.TestCase):

    def testAccessConnectionInAdapter(self):
        self.query("CREATE SCHEMA IF NOT EXISTS ADAPTER")
        self.query("CREATE CONNECTION my_conn TO 'MYADDRESS' USER 'MYUSER' IDENTIFIED BY 'MYPASSWORD'")
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 ADAPTER SCRIPT adapter.fast_adapter_conn AS
            import json
            import string
            encodeUTF8 = lambda x: x

            def adapter_call(request):
                root = json.loads(request)
                if root["type"] == "createVirtualSchema":
                    c = exa.get_connection("MY_CONN")
                    res = {
                        "type": "createVirtualSchema",
                        "schemaMetadata": {
                            "tables": [
                            {
                                "name": "T1",
                                "columns": [{
                                    "name": c.address,
                                    "dataType": {"type": "VARCHAR", "size": 2000000}
                                },{
                                    "name":  c.user,
                                    "dataType": {"type": "VARCHAR", "size": 2000000}
                                },{
                                    "name":  c.password,
                                    "dataType": {"type": "VARCHAR", "size": 2000000}
                                }]
                            }]
                        }
                    }
                    return encodeUTF8(json.dumps(res))
                elif root["type"] == "dropVirtualSchema":
                    return encodeUTF8(json.dumps({"type": "dropVirtualSchema"}))
                else:
                    raise ValueError('Unsupported callback')
            /
            '''))
        self.query("CREATE VIRTUAL SCHEMA VS USING ADAPTER.FAST_ADAPTER_CONN")
        rows = self.query("SELECT COLUMN_NAME FROM EXA_ALL_COLUMNS WHERE COLUMN_TABLE='T1' ORDER BY COLUMN_NAME")
        self.assertRowsEqual([('MYADDRESS',), ('MYPASSWORD',), ('MYUSER',)], rows)
        self.query("DROP FORCE VIRTUAL SCHEMA VS CASCADE")


class OptionalUSERandIDENTIFIEDBYTest(_Python3UdfSetup):

    def testNoUSERandNoIDENTIFIEDBY(self):
        self.query("CREATE or replace CONNECTION my_conn1 TO 'MYADDRESS'")
        rows = self.query('''SELECT fn1.print_connection('MY_CONN1')''')
        self.assertRowsEqual([('password', 'MYADDRESS', None, None)], rows)
        self.query("drop CONNECTION my_conn1")

    def testNoUSER(self):
        self.query("CREATE or replace CONNECTION my_conn2 TO 'MYADDRESS' identified by 'MYPASSWORD'")
        rows = self.query('''SELECT fn1.print_connection('MY_CONN2')''')
        self.assertRowsEqual([('password', 'MYADDRESS', None, 'MYPASSWORD')], rows)
        self.query("drop CONNECTION my_conn2")

    def testNoIDENIFIEDBY(self):
        self.query("CREATE or replace CONNECTION my_conn3 TO 'MYADDRESS' USER 'MYUSER'")
        rows = self.query('''SELECT fn1.print_connection('MY_CONN3')''')
        self.assertRowsEqual([('password', 'MYADDRESS', "MYUSER", None)], rows)
        self.query("drop CONNECTION my_conn3")


class GetConnectionAccessControlWithViewsTest(_Python3UdfSetup):
    def setUp(self):
        self._setup_common_udfs()
        
        self.query('''
        create connection AC_FOOCONN to 'a' user 'b' identified by 'c'
        ''', ignore_errors=True)

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid=username, pwd=password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username=username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username=username, password=password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

    def testUseConnectionUDFsInView(self):
        self.createUser("foo", "foo")
        self.query('create schema if not exists spot42542')
        self.query(
            "create or replace view spot42542.print_connection_wrapper as select fn1.print_connection('AC_FOOCONN')")
        self.query("grant select on spot42542.print_connection_wrapper to foo")
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        rows = foo_conn.query('''select * from spot42542.print_connection_wrapper''')
        foo_conn.commit()
        self.assertRowsEqual([('password', 'a', 'b', 'c')], rows)
        foo_conn.commit()
        self.query('drop schema spot42542 cascade')
        self.commit()


if __name__ == '__main__':
    udf.main()

