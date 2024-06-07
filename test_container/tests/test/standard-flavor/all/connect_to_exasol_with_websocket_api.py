#!/usr/bin/env python3
import json

from exasol_python_test_framework import udf


def db_port_from_environment_info_file():
    with open("/environment_info.json", "r") as f:
        env_info_dict = json.load(f)
        return env_info_dict["database_info"]["ports"]["database"]


class WebsocketAPIConnectionTest(udf.TestCase):
    # TODO use dsn and credentials injected into the testcase
    db_port = db_port_from_environment_info_file()
    connection = f"localhost:{db_port}"
    user = "sys"
    pwd = "exasol"

    def setUp(self):
        self.query('create schema websocket_api', ignore_errors=True)
    
    def run_secure_websocket_api_connection(self, python_version):
        self.query(udf.fixindent('''
            CREATE OR REPLACE %s SCALAR SCRIPT websocket_api.connect_secure() returns int AS
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
            ''' % (python_version, self.connection, self.user, self.pwd)))
        self.query('''SELECT websocket_api.connect_secure() FROM dual''')

    def test_secure_websocket_api_connection_python3(self):
        self.run_secure_websocket_api_connection("PYTHON3")

    def tearDown(self):
        self.query("drop schema websocket_api cascade")


if __name__ == '__main__':
    udf.main()
