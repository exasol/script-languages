#!/usr/bin/env python3

from exasol_python_test_framework import udf
import os


class WebsocketAPIConnectionTest(udf.TestCase):
    connection = "localhost:8888"
    user = "sys"
    pwd = "exasol"

    def setUp(self):
        self.clean()
        self.query('create schema row_level_security_data', ignore_errors=True)
        self.query('create schema row_level_security_adapter', ignore_errors=True)
        self.createUser("u1", "u1")
        self.createUser("u2", "u2")

        self.query(udf.fixindent('''            
            grant create session to u1;
        '''))
        self.query(udf.fixindent('''
            grant create session to u2;
        '''))
        self.query(udf.fixindent('''            
            create or replace table row_level_security_adapter.user_pref(username varchar(100), wants_only_active bool);
        '''))
        self.query(udf.fixindent('''            
            insert into
                row_level_security_adapter.user_pref
            values
                ('SYS', false),
                ('U1', true),
                ('U2', false);
        '''))
        self.query(udf.fixindent('''            
            CREATE OR REPLACE TABLE row_level_security_data.t(a1 varchar(100), a2 varchar(100), userName varchar(100), active bool);
        '''))
        self.query(udf.fixindent('''
            INSERT INTO
                row_level_security_data.t
            values
                ('a', 'b', 'SYS', true),
                ('c', 'd', 'SYS', false),
                ('e', 'f', 'U2', true),
                ('g', 'h', 'U2', false),
                ('i', 'j', 'U1', true),
                ('k', 'l', 'U1', false);
        '''))
        self.query(udf.fixindent('''
            create or replace connection sys_connection to 'wss://%s' user '%s' identified by '%s'
        ''' % (self.connection, "sys", "exasol")))
        self.query("commit")

    def test_row_level_security(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + "/row_level_security_adapter_script.txt", "r") as f:
            adapter_script = f.read()
        self.query(adapter_script)
        self.query(udf.fixindent('''
            CREATE VIRTUAL SCHEMA row_level_security 
            USING row_level_security_adapter.rls_adapter 
            with TABLE_SCHEMA='ROW_LEVEL_SECURITY_DATA'
                ADAPTER_SCHEMA='ROW_LEVEL_SECURITY_ADAPTER' 
                META_CONNECTION='SYS_CONNECTION';
        '''))
        self.query(udf.fixindent('''
            SELECT * FROM row_level_security.T;
        '''))
        self.query(udf.fixindent('''
            grant select on row_level_security to u1;
        '''))
        self.query(udf.fixindent('''
            grant select on row_level_security to u2;
        '''))
        self.query("commit")
        with self.expectations():
            self.check_data_access_for_user("u1", "u1")
            self.check_data_access_for_user("u2", "u2")
            self.check_row_level_security_for_user("u1", "u1", [("i", "j", True)])
            self.check_row_level_security_for_user("u2", "u2", [("e", "f", True), ("g", "h", False)])

    def check_data_access_for_user(self, user, pwd):
        data_access_failure = False
        try:
            con = self.getConnection(user, pwd)
            con.query("select * from row_level_security_data.t;")
        except:
            data_access_failure = True
        self.expectTrue(data_access_failure)

    def check_row_level_security_for_user(self, user, pwd, tuples):
        con = self.getConnection(user, pwd)
        rows = con.query("select * from row_level_security.t;")
        expected = [(t[0], t[1], user.upper(), t[2]) for t in tuples]
        self.expectRowsEqualIgnoreOrder(rows, expected)

    def tearDown(self):
        self.clean()

    def clean(self):
        self.query("drop schema row_level_security_data cascade", ignore_errors=True)
        self.query("drop schema row_level_security_adapter cascade", ignore_errors=True)
        self.query("drop force virtual schema row_level_security cascade;", ignore_errors=True)
        self.query("commit")


if __name__ == '__main__':
    udf.main()
