#!/usr/bin/env python3


from exasol_python_test_framework import udf


class PyexsolConnectionTest(udf.TestCase):
    # TODO use dsn and credentials injected into the testcase
    host = "localhost"
    port = "8888"
    user = "sys"
    pwd = "exasol"

    def setUp(self):
        self.query('create schema pyexasol', ignore_errors=True)
    
    def run_secure_pyexasol_connection(self, python_version):
        self.query(udf.fixindent('''
            CREATE OR REPLACE {python} SCALAR SCRIPT pyexasol.connect_secure() returns int AS
            import pyexasol
            import ssl
            import os
            def run(ctx):
                os.environ["USER"]="exasolution"
                with pyexasol.connect(
                        dsn='{host}:{port}', user='{user}', password='{pwd}', 
                        websocket_sslopt={{"cert_reqs": ssl.CERT_NONE}}, encryption=True) as connection:
                    result_set = connection.execute('SELECT 1 FROM dual')
                    for row in result_set:
                        pass
            /
            '''.format(python=python_version, host=self.host, port=self.port, user=self.user, pwd=self.pwd)))
        self.query('''SELECT pyexasol.connect_secure() FROM dual''')

    def run_fingerprint_pyexasol_connection(self, python_version):
        self.query(udf.fixindent('''
            CREATE OR REPLACE {python} SCALAR SCRIPT pyexasol.connect_secure() returns VARCHAR(2000000) AS
            import pyexasol
            import ssl
            import os
            def run(ctx):
                os.environ["USER"]="exasolution"
                try:
                    with pyexasol.connect(
                            dsn='{host}/135a1d2dce102de866f58267521f4232153545a075dc85f8f7596f57e588a181:{port}', 
                            user='{user}', password='{pwd}') as connection:
                        pass
                except pyexasol.ExaConnectionFailedError as e:
                    return e.message
            /
            '''.format(python=python_version, host=self.host, port=self.port, user=self.user, pwd=self.pwd)))
        rows=self.query('''SELECT pyexasol.connect_secure() FROM dual''')
        self.assertRegex(rows[0][0], r"Provided fingerprint.*did not match server fingerprint")

    def test_secure_pyexasol_connection_python3(self):
        self.run_secure_pyexasol_connection("PYTHON3")

    def test_fingerprint_pyexasol_connection_python3(self):
        self.run_fingerprint_pyexasol_connection("PYTHON3")

    def test_pyexasol_export_to_pandas(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT pyexasol.export_to_pandas() returns int AS
            import pyexasol
            import ssl
            import os
            def run(ctx):
                os.environ["USER"]="exasolution"
                with pyexasol.connect(
                        dsn='{host}:{port}', user='{user}', password='{pwd}', 
                        websocket_sslopt={{"cert_reqs": ssl.CERT_NONE}}, encryption=True) as connection:
                    result = connection.export_to_pandas('SELECT 1 FROM dual')
            /
            '''.format(host=self.host, port=self.port, user=self.user, pwd=self.pwd)))
        self.query('''SELECT pyexasol.export_to_pandas() FROM dual''')

    def tearDown(self):
        self.query("drop schema pyexasol cascade")


if __name__ == '__main__':
    udf.main()
