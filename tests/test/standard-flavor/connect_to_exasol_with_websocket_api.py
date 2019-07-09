#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf


class WebsocketAPIConnectionTest(udf.TestCase):
    connection = "localhost:8888"
    user = "sys"
    pwd = "exasol"

    def setUp(self):
        self.query('create schema websocket_api', ignore_errors=True)

    def test_unsecure_websocket_api_connection(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT websocket_api.connect_unsecure() returns int AS
            import EXASOL
            import os
            def run(ctx):
                os.environ["USER"]="exasolution"
                with EXASOL.connect('ws://%s', '%s', '%s') as connection:
                    with connection.cursor() as cursor:
                        cursor.execute('SELECT 1 FROM dual')
                        for row in cursor:
                            pass
            /
            ''' % (self.connection, self.user, self.pwd)))
        self.query('''SELECT websocket_api.connect_unsecure() FROM dual''')

    def test_secure_websocket_api_connection(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT websocket_api.connect_secure() returns int AS
            import EXASOL
            import ssl
            import os
            def run(ctx):
                os.environ["USER"]="exasolution"
                with EXASOL.connect('wss://%s', '%s', '%s', sslopt={"cert_reqs": ssl.CERT_NONE}) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute('SELECT 1 FROM dual')
                        for row in cursor:
                            pass
            /
            ''' % (self.connection, self.user, self.pwd)))
        self.query('''SELECT websocket_api.connect_secure() FROM dual''')

    def tearDown(self):
        self.query("drop schema websocket_api cascade")


if __name__ == '__main__':
    udf.main()

