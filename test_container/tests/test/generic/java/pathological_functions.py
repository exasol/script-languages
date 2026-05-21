#!/usr/bin/env python3

from exasol_python_test_framework import udf


class _JavaUdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            sleep("sec" double)
            RETURNS double AS
            class SLEEP {
                static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    int sec = ctx.getInteger("sec");
                    Thread.sleep(sec * 1000);
                    return sec;
                }
            }
            /
        '''))

class Test(_JavaUdfSetup):

    def test_query_timeout(self):
        self.query('ALTER SESSION SET QUERY_TIMEOUT = 10')
        try:
            with self.assertRaisesRegex(Exception, 'Successfully reconnected after query timeout'):
                self.query('SELECT fn1.sleep(100) FROM dual')
        finally:
            self.query('ALTER SESSION SET QUERY_TIMEOUT = 0')


if __name__ == '__main__':
    udf.main()
