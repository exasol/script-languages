#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework import exatest
from exasol_python_test_framework.udf import skip


class ImportAliasTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create or replace table fn2.t(z varchar(3000))')
        self.query('create or replace table fn2.t2(y varchar(2000), z varchar(3000))')
        self.query('''
                   create connection FOOCONN to 'a' user 'b' identified by 'c'
                   ''', ignore_errors=True)
        
        self.query('OPEN SCHEMA FN1')
        
        # Create all IMPORT UDF scripts
        self.query(udf.fixindent('''
            create or replace python3 set script impal_use_is_subselect(...) emits (x varchar(2000)) as
            def generate_sql_for_import_spec(import_spec):
                    return "select " + str(import_spec.is_subselect)
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 scalar script impal_use_param_foo_bar(...) returns varchar(2000) as
            def generate_sql_for_import_spec(import_spec):
                    return "select '" +  import_spec.parameters['FOO']  + "', '" + import_spec.parameters['BAR'] + "'"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 set script impal_use_connection_name(...) emits (x varchar(2000)) as
            def generate_sql_for_import_spec(import_spec):
                return "select '" + import_spec.connection_name + "'"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 set script impal_use_connection_fooconn(...) emits (x varchar(2000)) as
            def generate_sql_for_import_spec(import_spec):
                c = exa.get_connection('FOOCONN')
                return "select '" + str(c.address) + str(c.user) + str(c.password) + "'"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 set script impal_use_connection(...) emits (x varchar(2000)) as
            def generate_sql_for_import_spec(import_spec):
                    return "select '" + import_spec.connection.user + import_spec.connection.password + import_spec.connection.address + import_spec.connection.type + "'"
            /
        '''))
        
        self.query(udf.fixindent('''
            create or replace python3 set script impal_use_all(...) emits (x varchar(2000)) as
            def generate_sql_for_import_spec(import_spec):
                    is_sub = str(import_spec.is_subselect).upper()
                    connection_string = 'X'
                    connection_name = 'Y'
                    foo = 'Z'
                    types = 'T'
                    names = 'N'
                    if import_spec.connection is not None:
                            connection_string = import_spec.connection.user + import_spec.connection.password + import_spec.connection.address + import_spec.connection.type
                    if import_spec.connection_name is not None:
                            connection_name = import_spec.connection_name
                    if import_spec.parameters['FOO'] is not None:
                            foo = import_spec.parameters['FOO']
                    if import_spec.subselect_column_types is not None:
                            for i in range(0, len(import_spec.subselect_column_types)):
                                    types = types + import_spec.subselect_column_types[i]
                                    names = names + import_spec.subselect_column_names[i]
                    return "select 1, '" + is_sub + '_' + connection_name + '_' + connection_string + '_' +  foo + '_' + types + '_' + names + "'"
            /
        '''))

    def test_import_use_is_subselect(self):
        self.query('''
            IMPORT INTO fn2.t FROM SCRIPT fn1.impal_use_is_subselect
            ''')
        rows = self.query('select * from fn2.t')
        self.assertRowsEqual([('FALSE',)], rows)
        self.query('truncate table fn2.t')

    def test_import_use_is_subselect_subselect(self):
        rows = self.query('''
            SELECT * FROM (IMPORT FROM SCRIPT fn1.impal_use_is_subselect)
            ''')
        self.assertRowsEqual([(True,)], rows)
        rows = self.query('''
            IMPORT FROM SCRIPT fn1.impal_use_is_subselect
            ''')
        self.assertRowsEqual([(True,)], rows)

    def test_import_use_params(self):
        self.query("IMPORT INTO fn2.t2 FROM SCRIPT fn1.impal_use_param_foo_bar with foo='bar' bar='foo'")
        rows = self.query('select * from fn2.t2')
        self.assertRowsEqual([('bar','foo')], rows)
        self.query('truncate table fn2.t2')

    def test_import_use_params_subselect(self):
        rows = self.query("SELECT * FROM (IMPORT FROM SCRIPT fn1.impal_use_param_foo_bar with foo='bar' bar='foo')")
        self.assertRowsEqual([('bar','foo')], rows)
        rows = self.query("IMPORT FROM SCRIPT fn1.impal_use_param_foo_bar with foo='bar' bar='foo'")
        self.assertRowsEqual([('bar','foo')], rows)

    def test_import_use_connection_name(self):
        self.query('IMPORT INTO fn2.t FROM SCRIPT fn1.impal_use_connection_name at fooconn')
        rows = self.query('select * from fn2.t')
        self.assertRowsEqual([('FOOCONN',)], rows)
        self.query('truncate table fn2.t')

    def test_import_use_connection_fooconn(self):
        rows = self.query('IMPORT FROM SCRIPT fn1.impal_use_connection_fooconn')
        self.assertRowsEqual([('abc',)], rows)
        self.query('truncate table fn2.t')

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid = username, pwd = password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username = username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username = username, password = password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

    def test_import_use_connection_fooconn_fails_for_user_foo(self):
        self.createUser('foo','foo')
        self.commit()
        foo_conn = self.getConnection('foo','foo')
        with self.assertRaisesRegex(Exception, 'insufficient privileges'):
            foo_conn.query('IMPORT FROM SCRIPT fn1.impal_use_connection_fooconn')
        self.query('drop user foo cascade')

    @skip("IMPORT FROM SCRIPT cannot be used in view definitions")
    def test_import_use_connection_fooconn_for_user_foo_and_view(self):
        self.query('create view fn2.fooconn_import_view as IMPORT FROM SCRIPT fn1.impal_use_connection_fooconn')
        self.createUser('foo','foo')
        self.commit()
        foo_conn = self.getConnection('foo','foo')
        rows = foo_conn.query('select * from fn2.fooconn_import_view')
        self.assertRowsEqual([('abc',)], rows)
        self.query('drop user foo cascade')
        self.query('drop view fn2.fooconn_import_view')

    def test_import_use_connection_name_subselect(self):
        rows = self.query('SELECT * FROM (IMPORT FROM SCRIPT fn1.impal_use_connection_name at fooconn)')
        self.assertRowsEqual([('FOOCONN',)], rows)
        rows = self.query('IMPORT FROM SCRIPT fn1.impal_use_connection_name at fooconn')
        self.assertRowsEqual([('FOOCONN',)], rows)

    def test_import_use_connection(self):
        self.query('''
            IMPORT INTO fn2.t FROM SCRIPT fn1.impal_use_connection
            at 'fooconn' user 'hans' identified by 'meiser'
            ''')
        rows = self.query('select * from fn2.t')
        self.assertRowsEqual([('hansmeiserfooconnpassword',)], rows)
        self.query('truncate table fn2.t')

    def test_import_use_connection_subselect(self):
        rows = self.query(''' SELECT * FROM (
            IMPORT FROM SCRIPT fn1.impal_use_connection
            at 'fooconn' user 'hans' identified by 'meiser')
            ''')
        self.assertRowsEqual([('hansmeiserfooconnpassword',)], rows)
        rows = self.query('''
            IMPORT FROM SCRIPT fn1.impal_use_connection
            at 'fooconn' user 'hans' identified by 'meiser'
            ''')
        self.assertRowsEqual([('hansmeiserfooconnpassword',)], rows)

    def test_import_use_all(self):
        self.query('''
            IMPORT INTO fn2.t2 FROM SCRIPT fn1.impal_use_all
            at 'fooconn' user 'hans' identified by 'meiser' with foo='a value'
            ''')
        rows = self.query('select * from fn2.t2')
        self.assertRowsEqual([('1','FALSE_Y_hansmeiserfooconnpassword_a value_T_N')], rows)
        self.query('truncate table fn2.t2')

    def test_import_use_all_subselect(self):
        rows = self.query(''' SELECT * FROM (
            IMPORT INTO (a double, b varchar(3000)) FROM SCRIPT fn1.impal_use_all
            at 'fooconn' user 'hans' identified by 'meiser' with foo='a value')
            ''')
        self.assertRowsEqual([(1, 'TRUE_Y_hansmeiserfooconnpassword_a value_TDOUBLEVARCHAR(3000) UTF8_NAB')], rows)
        rows = self.query('''
        IMPORT INTO (a double, b varchar(3000)) FROM SCRIPT fn1.impal_use_all
        at 'fooconn' user 'hans' identified by 'meiser' with foo='a value'
            ''')
        self.assertRowsEqual([(1, 'TRUE_Y_hansmeiserfooconnpassword_a value_TDOUBLEVARCHAR(3000) UTF8_NAB')], rows)

    def test_prepared_statement_params(self):
        with self.assertRaisesRegex(Exception, 'syntax error, unexpected \'?\''):
            rows = self.query(''' SELECT * FROM (
                IMPORT INTO (a double, b varchar(3000)) FROM SCRIPT fn1.impal_use_all
                at 'fooconn' user 'hans' identified by 'meiser' with foo=?)
                ''', 'bar')

    def test_prepared_statement_conn(self):
        with self.assertRaisesRegex(Exception, 'syntax error, unexpected \'?\''):
            rows = self.query(''' SELECT * FROM (
                IMPORT INTO (a double, b varchar(3000)) FROM SCRIPT fn1.impal_use_all
                at ? user ? identified by ? with foo='bar')
                ''', 'fooconn', 'hans', 'meiser', 'bar')

    def test_import_in_lua_scripting(self):
        self.query('''
            create or replace script s1() as
                res = pquery [[ IMPORT INTO fn2.t FROM SCRIPT fn1.impal_use_is_subselect ]]
         ''')
        self.query('execute script s1()')
        rows = self.query('select * from fn2.t')
        self.assertRowsEqual([('FALSE',)], rows)
        self.query('truncate table fn2.t')


if __name__ == '__main__':
    udf.main()
