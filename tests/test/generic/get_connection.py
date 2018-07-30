#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import requires
import exatest


class AccessConnectionSysPriv(udf.TestCase):
    def testSysPrivExists(self):
        sysPriv = self.query("SELECT * FROM EXA_DBA_SYS_PRIVS WHERE PRIVILEGE = 'ACCESS ANY CONNECTION'")
        self.assertRowsEqual([("DBA", "ACCESS ANY CONNECTION", True)], sysPriv)

class GetConnectionTest(udf.TestCase):
    def setUp(self):
        self.query('''
        create connection FOOCONN to 'a' user 'b' identified by 'c'
        ''')

    def tearDown(self):
        self.query('drop connection FOOCONN')

    @requires('PRINT_CONNECTION')
    def test_print_existing_connection(self):
        rows = self.query('''
            SELECT fn1.print_connection('FOOCONN')
            ''')
        self.assertRowsEqual([('password','a','b','c')], rows)

    @requires('PRINT_CONNECTION')
    def test_connection_not_found(self):
        with self.assertRaisesRegexp(Exception, 'connection FOO does not exist'):
            self.query('''
                SELECT fn1.print_connection('FOO')
                ''')

class GetConnectionAccessControlTest(udf.TestCase):
    def setUp(self):
        self.query('''
        create connection AC_FOOCONN to 'a' user 'b' identified by 'c'
        ''', ignore_errors=True)

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid = username, pwd = password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username = username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username = username, password = password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

    def testUseConnectionWithoutRights(self):
        self.createUser("foo", "foo")
        self.query('grant create schema to foo')
        self.query('grant create script to foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        foo_conn.query('create schema foos')
        #lua
        foo_conn.query('''
            create or replace lua scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            function run(ctx)
              local c = exa.get_connection(ctx.conn)
              ctx.emit( c.type,  c.address,  c.user,  c.password )
            end
         ''')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                select print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        #python
        foo_conn.query('''
            create or replace python scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            def run(ctx):
                    c = exa.get_connection(ctx.conn)
                    ctx.emit( c.type,  c.address,  c.user,  c.password )
         ''')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                select print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        #r
        foo_conn.query('''
            create or replace r scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            run <- function(ctx) {
                c = exa$get_connection(ctx$conn)
                ctx$emit( c$type,  c$address,  c$user,  c$password )
        }
         ''')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                select print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        #java
        foo_conn.query('''
            CREATE or replace java SCALAR SCRIPT print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
                %jvmoption -Xms64m -Xmx128m -Xss512k;
                class PRINT_CONNECTION {
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        ExaConnectionInformation c = exa.getConnection(ctx.getString("conn"));
                        ctx.emit(c.getType().toString().toLowerCase(),c.getAddress(),c.getUser(), c.getPassword());
                    }
                }
         ''')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                select print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        self.query('drop user foo cascade')
        self.commit()

    def testUseConnectionWithOldRight(self):
        self.createUser("foo", "foo")
        self.query('grant create schema to foo')
        self.query('grant create script to foo')
        self.query('grant connection ac_fooconn to foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        foo_conn.query('create schema foos')
        #lua
        foo_conn.query('''
            create or replace lua scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            function run(ctx)
              local c = exa.get_connection(ctx.conn)
              ctx.emit( c.type,  c.address,  c.user,  c.password )
            end
         ''')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                select print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        #python
        foo_conn.query('''
            create or replace python scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            def run(ctx):
                    c = exa.get_connection(ctx.conn)
                    ctx.emit( c.type,  c.address,  c.user,  c.password )
         ''')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                select print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        #r
        foo_conn.query('''
            create or replace r scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            run <- function(ctx) {
                c = exa$get_connection(ctx$conn)
                ctx$emit( c$type,  c$address,  c$user,  c$password )
        }
         ''')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                select print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        #java
        foo_conn.query('''
            CREATE or replace java SCALAR SCRIPT print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
                %jvmoption -Xms64m -Xmx128m -Xss512k;
                class PRINT_CONNECTION {
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        ExaConnectionInformation c = exa.getConnection(ctx.getString("conn"));
                        ctx.emit(c.getType().toString().toLowerCase(),c.getAddress(),c.getUser(), c.getPassword());
                    }
                }
         ''')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection AC_FOOCONN in script PRINT_CONNECTION'):
            foo_conn.query('''
                select print_connection('AC_FOOCONN')
            ''')
        foo_conn.commit()
        self.query('drop user foo cascade')
        self.commit()


    def testUseConnectionWithNewRight(self):
        self.createUser("foo", "foo")
        self.query('grant create schema to foo')
        self.query('grant create script to foo')
        self.query('GRANT ACCESS ON CONNECTION ac_fooconn to foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        foo_conn.query('create schema foos')
        #lua
        foo_conn.query('''
            create or replace lua scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            function run(ctx)
              local c = exa.get_connection(ctx.conn)
              ctx.emit( c.type,  c.address,  c.user,  c.password )
            end
         ''')
        rows = foo_conn.query('''
           select print_connection('AC_FOOCONN')
        ''')
        self.assertRowsEqual([('password','a','b','c')], rows)
        foo_conn.commit()
        #python
        foo_conn.query('''
            create or replace python scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            def run(ctx):
                    c = exa.get_connection(ctx.conn)
                    ctx.emit( c.type,  c.address,  c.user,  c.password )
         ''')
        rows = foo_conn.query('''
            select print_connection('AC_FOOCONN')
        ''')
        self.assertRowsEqual([('password','a','b','c')], rows)
        foo_conn.commit()
        #r
        foo_conn.query('''
            create or replace r scalar script print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            run <- function(ctx) {
                c = exa$get_connection(ctx$conn)
                ctx$emit( c$type,  c$address,  c$user,  c$password )
        }
         ''')
        rows = foo_conn.query('''
             select print_connection('AC_FOOCONN')
        ''')
        self.assertRowsEqual([('password','a','b','c')], rows)
        foo_conn.commit()
        #java
        foo_conn.query('''
            CREATE or replace java SCALAR SCRIPT print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
                %jvmoption -Xms64m -Xmx128m -Xss512k;
                class PRINT_CONNECTION {
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        ExaConnectionInformation c = exa.getConnection(ctx.getString("conn"));
                        ctx.emit(c.getType().toString().toLowerCase(),c.getAddress(),c.getUser(), c.getPassword());
                    }
                }
         ''')
        rows = foo_conn.query('''
             select print_connection('AC_FOOCONN')
        ''')
        self.assertRowsEqual([('password','a','b','c')], rows)
        foo_conn.commit()
        self.query('drop user foo cascade')
        self.commit()

    def testUseConnectionInOldImportWithNewRight(self):
        self.createUser("foo", "foo")
        self.query('grant insert any table to foo')
        self.query('GRANT ACCESS ON CONNECTION ac_fooconn to foo')
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for using connection'):
            foo_conn.query('''
                import from fbv at ac_fooconn file 'foo'
            ''')
        foo_conn.commit()
        self.query('drop user foo cascade')
        self.commit()

class BigConnectionTest(udf.TestCase):

    # Should be max. size 2.000.000, but this will cause our odbc driver to crash (sigsegv) during logging (DWA-20290). Will be increased to max size when bug is fixed
    address = "a" * 2 * 1000 * 100
    user = "u" * 2 * 1000 * 100
    password = "p" * 2 * 1000 * 100

    def setUp(self):
        self.query('''
            create connection LARGEST_CONN to '{address}' user '{user}' identified by '{password}'
            '''.format(address = self.address, user = self.user, password = self.password))

    def tearDown(self):
        self.query("DROP CONNECTION LARGEST_CONN")

    @requires('PRINT_CONNECTION')
    def testGetBigConnection(self):
        rows = self.query('''
            SELECT fn1.print_connection('LARGEST_CONN')
            ''')
        self.assertRowsEqual([('password', self.address, self.user, self.password)], rows)

    def testBigConnectionSysTables(self):
        rows = self.query('''
            SELECT CONNECTION_STRING, USER_NAME, PASSWORD FROM "$EXA_DBA_CONNECTIONS" WHERE CONNECTION_NAME = 'LARGEST_CONN'
            ''')
        self.assertRowsEqual([(self.address, self.user, self.password)], rows)
        rows = self.query('''
            SELECT CONNECTION_STRING, USER_NAME FROM EXA_DBA_CONNECTIONS WHERE CONNECTION_NAME = 'LARGEST_CONN'
            ''')
        self.assertRowsEqual([(self.address, self.user)], rows)
        res = self.query('''
            SELECT count(*) FROM EXA_ALL_CONNECTIONS WHERE CONNECTION_NAME = 'LARGEST_CONN'
            ''')
        self.assertRowsEqual([(1, )], res)

class ConnectionTest(udf.TestCase):

    def testAccessConnectionInAdapter(self):
        self.query("CREATE SCHEMA IF NOT EXISTS ADAPTER")
        self.query("CREATE CONNECTION my_conn TO 'MYADDRESS' USER 'MYUSER' IDENTIFIED BY 'MYPASSWORD'");
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT adapter.fast_adapter_conn AS
            import json
            import string
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
                    return json.dumps(res).encode('utf-8')
                elif root["type"] == "dropVirtualSchema":
                    return json.dumps({"type": "dropVirtualSchema"}).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            '''))
        self.query("CREATE VIRTUAL SCHEMA VS USING ADAPTER.FAST_ADAPTER_CONN")
        rows = self.query("SELECT COLUMN_NAME FROM EXA_ALL_COLUMNS WHERE COLUMN_TABLE='T1' ORDER BY COLUMN_NAME")
        self.assertRowsEqual([('MYADDRESS',), ('MYPASSWORD',), ('MYUSER',)], rows)
        self.query("DROP FORCE VIRTUAL SCHEMA VS CASCADE")



class OptionalUSERandIDENTIFIEDBYTest(udf.TestCase):

    @requires('PRINT_CONNECTION')
    def testNoUSERandNoIDENTIFIEDBY(self):
        self.query("CREATE or replace CONNECTION my_conn1 TO 'MYADDRESS'")
        rows = self.query('''SELECT fn1.print_connection('MY_CONN1')''')
        self.assertRowsEqual([('password', 'MYADDRESS', None, None)], rows)
        self.query("drop CONNECTION my_conn1");

    @requires('PRINT_CONNECTION')
    def testNoUSER(self):
        self.query("CREATE or replace CONNECTION my_conn2 TO 'MYADDRESS' identified by 'MYPASSWORD'")
        rows = self.query('''SELECT fn1.print_connection('MY_CONN2')''')
        self.assertRowsEqual([('password', 'MYADDRESS', None, 'MYPASSWORD')], rows)
        self.query("drop CONNECTION my_conn2");

    @requires('PRINT_CONNECTION')
    def testNoIDENIFIEDBY(self):
        self.query("CREATE or replace CONNECTION my_conn3 TO 'MYADDRESS' USER 'MYUSER'")
        rows = self.query('''SELECT fn1.print_connection('MY_CONN3')''')
        self.assertRowsEqual([('password', 'MYADDRESS', "MYUSER", None)], rows)
        self.query("drop CONNECTION my_conn3");



##
##
##
##

class GetConnectionAccessControlWithViewsTest(udf.TestCase):
    def setUp(self):
        self.query('''
        create connection AC_FOOCONN to 'a' user 'b' identified by 'c'
        ''', ignore_errors=True)

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid = username, pwd = password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username = username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username = username, password = password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))


    def testUseConnectionUDFsInView(self):
        self.createUser("foo", "foo")
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')

        #lua
        self.query('create schema if not exists spot42542_lua')
        self.query(udf.fixindent('''
            create or replace lua scalar script spot42542_lua.print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            function run(ctx)
              local c = exa.get_connection(ctx.conn)
              ctx.emit( c.type,  c.address,  c.user,  c.password )
            end
         '''))
        self.commit()
        self.query("create or replace view spot42542_lua.print_connection_wrapper as select spot42542_lua.print_connection('AC_FOOCONN')")
        self.query("grant select on spot42542_lua.print_connection_wrapper to foo")
        self.commit()
        rows = foo_conn.query('''select * from spot42542_lua.print_connection_wrapper''')
        foo_conn.commit()
        self.assertRowsEqual([('password','a','b','c')], rows)
        self.query('drop schema spot42542_lua cascade')
        self.commit()

        #python
        self.query('create schema if not exists spot42542_python')
        self.query(udf.fixindent('''
            create or replace python scalar script spot42542_python.print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            def run(ctx):
                    c = exa.get_connection(ctx.conn)
                    ctx.emit( c.type,  c.address,  c.user,  c.password )
         '''))
        self.commit()
        self.query("create or replace view spot42542_python.print_connection_wrapper as select spot42542_python.print_connection('AC_FOOCONN')")
        self.query("grant select on spot42542_python.print_connection_wrapper to foo")
        self.commit()
        rows = foo_conn.query('''select * from spot42542_python.print_connection_wrapper''')
        foo_conn.commit()
        self.assertRowsEqual([('password','a','b','c')], rows)
        self.query('drop schema spot42542_python cascade')
        self.commit()

        #r
        self.query('create schema if not exists spot42542_r')
        self.query(udf.fixindent('''
            create or replace r scalar script spot42542_r.print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
            run <- function(ctx) {
                c = exa$get_connection(ctx$conn)
                ctx$emit( c$type,  c$address,  c$user,  c$password )
        }
        '''))
        self.commit()
        self.query("create or replace view spot42542_r.print_connection_wrapper as select spot42542_r.print_connection('AC_FOOCONN')")
        self.query("grant select on spot42542_r.print_connection_wrapper to foo")
        self.commit()
        rows = foo_conn.query('''select * from spot42542_r.print_connection_wrapper''')
        foo_conn.commit()
        self.assertRowsEqual([('password','a','b','c')], rows)
        self.query('drop schema spot42542_r cascade')
        self.commit()

        #java
        self.query('create schema if not exists spot42542_java')
        self.query(udf.fixindent('''
            CREATE or replace java SCALAR SCRIPT spot42542_java.print_connection(conn varchar (1000))
            emits(type varchar(200), host varchar(200), conn varchar(200), pwd varchar(200))
            as
                %jvmoption -Xms64m -Xmx128m -Xss512k;
                class PRINT_CONNECTION {
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        ExaConnectionInformation c = exa.getConnection(ctx.getString("conn"));
                        ctx.emit(c.getType().toString().toLowerCase(),c.getAddress(),c.getUser(), c.getPassword());
                    }
                }
        '''))
        self.commit()
        self.query("create or replace view spot42542_java.print_connection_wrapper as select spot42542_java.print_connection('AC_FOOCONN')")
        self.query("grant select on spot42542_java.print_connection_wrapper to foo")
        self.commit()
        rows = foo_conn.query('''select * from spot42542_java.print_connection_wrapper''')
        foo_conn.commit()
        self.assertRowsEqual([('password','a','b','c')], rows)
        self.query('drop schema spot42542_java cascade')
        self.commit()



if __name__ == '__main__':
    udf.main()
