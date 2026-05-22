#!/usr/bin/env python3

from exasol_python_test_framework import udf


class Test(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT sleep("sec" double)
            RETURNS double AS
            import time
            
            def run(ctx):
                time.sleep(ctx.sec)
                return ctx.sec
            /
        '''))

    def test_query_timeout(self):
        self.query('ALTER SESSION SET QUERY_TIMEOUT = 10')
        try:
            with self.assertRaisesRegex(Exception, 'Successfully reconnected after query timeout'):
                self.query('SELECT fn1.sleep(100) FROM dual')
        finally:
            self.query('ALTER SESSION SET QUERY_TIMEOUT = 0')


if __name__ == '__main__':
    udf.main()

