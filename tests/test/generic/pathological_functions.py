#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires


class Test(udf.TestCase):

    @requires('SLEEP')
    def test_query_timeout(self):
        self.query('ALTER SESSION SET QUERY_TIMEOUT = 10')
        try:
            with self.assertRaisesRegex(Exception, 'Successfully reconnected after query timeout'):
                self.query('SELECT fn1.sleep(100) FROM dual')
        finally:
            self.query('ALTER SESSION SET QUERY_TIMEOUT = 0')


if __name__ == '__main__':
    udf.main()
