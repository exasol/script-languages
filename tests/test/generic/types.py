#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import requires, expectedFailureIfLang

class TestEcho(udf.TestCase):

    @requires('ECHO_BOOLEAN')
    def test_echo_boolean(self):
        rows = self.query('''
            SELECT
                fn1.echo_boolean(Null) is NULL,
                fn1.echo_boolean(True) = True,
                fn1.echo_boolean(False) = False
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True)], rows)

    @requires('ECHO_CHAR1')
    def test_echo_char1(self):
        rows = self.query('''
            SELECT
                fn1.echo_char1(NULL) is NULL,
                fn1.echo_char1('a') = 'a'
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    @requires('ECHO_CHAR10')
    def test_echo_char10(self):
        rows = self.query('''
            SELECT
                fn1.echo_char10(NULL) is NULL,
                fn1.echo_char10('ab') = 'ab        '
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    @requires('ECHO_DATE')
    def test_echo_date(self):
        rows = self.query('''
            SELECT
                fn1.echo_date(NULL) is NULL,
                fn1.echo_date(current_date()) = current_date()
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    @requires('ECHO_INTEGER')
    def test_echo_integer_basic(self):
        rows = self.query('''
            SELECT
                fn1.echo_integer(NULL) is NULL,
                fn1.echo_integer(-1) = -1,
                fn1.echo_integer(0) = 0,
                fn1.echo_integer(1) = 1
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True, True)], rows)

    @requires('ECHO_INTEGER')
    @expectedFailureIfLang('r')
    def test_echo_integer_limits(self):
        '''DWA-13784 (R)'''
        rows = self.query('''
            SELECT
                fn1.echo_integer(-(1e18 - 1)) = -(1e18 - 1),
                fn1.echo_integer(  1e18 - 1)  =   1e18 - 1
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    @requires('ECHO_DOUBLE')
    def test_echo_double(self):
        rows = self.query('''
            SELECT
                fn1.echo_double(NULL) is NULL,
                fn1.echo_double(CAST(1.5 AS DOUBLE)) = CAST(1.5 AS DOUBLE),
                fn1.echo_double(0) = 0.0,
                fn1.echo_double(0.0) = 0.0,
                fn1.echo_double(-1.7e-308) = -1.7e-308,
                fn1.echo_double(+1.7e-308) = +1.7e-308
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True, True, True, True)], rows)

    @requires('ECHO_DECIMAL_36_0')
    def test_echo_decimal_36_0_basic(self):
        rows = self.query('''
            SELECT
                fn1.echo_decimal_36_0(NULL) is NULL,
                fn1.echo_decimal_36_0(0) = 0,
                fn1.echo_decimal_36_0(0.0) = 0.0
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True)], rows)

    @requires('ECHO_DECIMAL_36_0')
    @expectedFailureIfLang('r')
    def test_echo_decimal_36_0_limits(self):
        '''DWA-13784 (R)'''
        rows = self.query('''
            SELECT
                fn1.echo_decimal_36_0(-(1e35 - 1)) = -(1e35 - 1),
                fn1.echo_decimal_36_0(  1e35 - 1)  =   1e35 - 1
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    @requires('ECHO_DECIMAL_36_36')
    def test_echo_decimal_36_36_basic(self):
        rows = self.query('''
            SELECT
                fn1.echo_decimal_36_36(NULL) is NULL,
                fn1.echo_decimal_36_36(0) = 0,
                fn1.echo_decimal_36_36(0.0) = 0.0
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True)], rows)

    @requires('ECHO_DECIMAL_36_36')
    @expectedFailureIfLang('r')
    def test_echo_decimal_36_36_limits(self):
        '''DWA-13784 (R)'''
        rows = self.query('''
            SELECT
                fn1.echo_decimal_36_36(-(1e-35 - 1)) = -(1e-35 - 1),
                fn1.echo_decimal_36_36(  1e-35 - 1)  =   1e-35 - 1
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    @requires('ECHO_VARCHAR10')
    def test_echo_varchar10(self):
        rows = self.query('''
            SELECT
                fn1.echo_varchar10(NULL) is NULL,
                fn1.echo_varchar10('') is NULL,
                fn1.echo_varchar10(' ') = ' ',
                fn1.echo_varchar10('a') = 'a',
                fn1.echo_varchar10('a ') = 'a ',
                fn1.echo_varchar10(' a ') = ' a '
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True, True, True, True)], rows)

    @requires('ECHO_TIMESTAMP')
    def test_echo_timestamp(self):
        rows = self.query('''
            SELECT fn1.echo_timestamp(NULL) is NULL
            FROM DUAL''')
        self.assertRowsEqual([(True,)], rows)
        
        rows = self.query('''
            SELECT fn1.echo_timestamp('2017-08-01 13:13:50.910') = '2017-08-01 13:13:50.910',
                   fn1.echo_timestamp('2017-08-01 13:13:50.983') = '2017-08-01 13:13:50.983'
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

        rows = self.query('''
            SELECT x1 = x2, x1, x2, x3
            FROM (SELECT fn1.echo_timestamp(x) AS x1, x AS x2, x AS x3
                  FROM (SELECT now() AS x FROM DUAL))''')
        self.assertEqual(True, rows[0][0], str(rows))


class EmptyTest(udf.TestCase):
    @requires('RUN_FUNC_IS_EMPTY')
    def test_run_func_is_empty(self):
        rows = self.query('''
            SELECT
                fn1.run_func_is_empty() IS NULL
            FROM DUAL''')
        self.assertRowsEqual([(True,)], rows)


class BottleneckTest(udf.TestCase):
    @requires('BOTTLENECK_VARCHAR10')
    def test_varchar10(self):
        for i in 0, 1, 5, 10:
            rows = self.query('''
                SELECT fn1.bottleneck_varchar10('%s')
                FROM DUAL''' % ('x' * i))
            self.assertEqual('x' * i if i > 0 else None, rows[0][0])
        with self.assertRaises(Exception):
            self.query('''
                SELECT fn1.bottleneck_varchar10('%s')
                FROM DUAL''' % ('x' * 11))

    @requires('BOTTLENECK_CHAR10')
    def test_char10(self):
        for i in 0, 1, 5, 10:
            rows = self.query('''
                SELECT fn1.bottleneck_char10('%s')
                FROM DUAL''' % ('x' * i))
            self.assertEqual(
                    ('x' * i + ' ' * 10)[:10] if i > 0 else None,
                    rows[0][0])
        with self.assertRaises(Exception):
            self.query('''
                SELECT fn1.bottleneck_char10('%s')
                FROM DUAL''' % ('x' * 11))

    @requires('BOTTLENECK_DECIMAL5')
    def test_decimal5(self):
        for i in 3, 4:
            rows = self.query('''
                SELECT fn1.bottleneck_decimal5(%d)
                FROM DUAL''' % (10 ** i))
            self.assertEqual(10 ** i, rows[0][0])
        with self.assertRaises(Exception):
            self.query('''
                SELECT fn1.bottleneck_decimal5(%d)
                FROM DUAL''' % (10 ** 5))

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
