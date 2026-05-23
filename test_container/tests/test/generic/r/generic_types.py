#!/usr/bin/env python3

from exasol_python_test_framework import udf
import pathlib


class TestEcho(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Load R scripts from SQL file
        sql_file = pathlib.Path(__file__).parent.parent.parent.parent / 'lang' / 'r' / 'types.sql'
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        # Execute each CREATE SCRIPT statement
        statements = sql_content.split('/')
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and 'CREATE' in stmt.upper():
                self.query(stmt)


    def test_echo_boolean(self):
        rows = self.query('''
            SELECT
                fn1.echo_boolean(Null) is NULL,
                fn1.echo_boolean(True) = True,
                fn1.echo_boolean(False) = False
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True)], rows)

    def test_echo_char1(self):
        rows = self.query('''
            SELECT
                fn1.echo_char1(NULL) is NULL,
                fn1.echo_char1('a') = 'a'
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    def test_echo_char10(self):
        rows = self.query('''
            SELECT
                fn1.echo_char10(NULL) is NULL,
                fn1.echo_char10('ab') = 'ab        '
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    def test_echo_date(self):
        rows = self.query('''
            SELECT
                fn1.echo_date(NULL) is NULL,
                fn1.echo_date(current_date()) = current_date()
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    def test_echo_integer_basic(self):
        rows = self.query('''
            SELECT
                fn1.echo_integer(NULL) is NULL,
                fn1.echo_integer(-1) = -1,
                fn1.echo_integer(0) = 0,
                fn1.echo_integer(1) = 1
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True, True)], rows)

    @udf.TestCase.expectedFailureIfLang('r')
    def test_echo_integer_limits(self):
        """DWA-13784 (R)"""
        rows = self.query('''
            SELECT
                fn1.echo_integer(-(1e18 - 1)) = -(1e18 - 1),
                fn1.echo_integer(  1e18 - 1)  =   1e18 - 1
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

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

    def test_echo_decimal_36_0_basic(self):
        rows = self.query('''
            SELECT
                fn1.echo_decimal_36_0(NULL) is NULL,
                fn1.echo_decimal_36_0(0) = 0,
                fn1.echo_decimal_36_0(0.0) = 0.0
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True)], rows)

    @udf.TestCase.expectedFailureIfLang('r')
    def test_echo_decimal_36_0_limits(self):
        """DWA-13784 (R)"""
        rows = self.query('''
            SELECT
                fn1.echo_decimal_36_0(-(1e35 - 1)) = -(1e35 - 1),
                fn1.echo_decimal_36_0(  1e35 - 1)  =   1e35 - 1
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

    def test_echo_decimal_36_36_basic(self):
        rows = self.query('''
            SELECT
                fn1.echo_decimal_36_36(NULL) is NULL,
                fn1.echo_decimal_36_36(0) = 0,
                fn1.echo_decimal_36_36(0.0) = 0.0
            FROM DUAL''')
        self.assertRowsEqual([(True, True, True)], rows)

    @udf.TestCase.expectedFailureIfLang('r')
    def test_echo_decimal_36_36_limits(self):
        """DWA-13784 (R)"""
        rows = self.query('''
            SELECT
                fn1.echo_decimal_36_36(-(1e-35 - 1)) = -(1e-35 - 1),
                fn1.echo_decimal_36_36(  1e-35 - 1)  =   1e-35 - 1
            FROM DUAL''')
        self.assertRowsEqual([(True, True)], rows)

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

        self.query('DROP SCHEMA ECHO_TEST CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA ECHO_TEST')
        self.query('''CREATE OR REPLACE TABLE N1 (now VARCHAR(255))''')
        self.query('''INSERT INTO N1 VALUES (SELECT now() AS x FROM DUAL)''')
        self.query('''    
            SELECT x1 = x2, x1, x2, x3
            FROM (SELECT fn1.echo_timestamp(x) AS x1, x AS x2, x AS x3
                  FROM (SELECT now AS x FROM N1))
                  ''')
        self.assertEqual(True, rows[0][0], str(rows))
        self.query('''DROP SCHEMA ECHO_TEST CASCADE''')


class EmptyTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Load R scripts from SQL file
        sql_file = pathlib.Path(__file__).parent.parent.parent.parent / 'lang' / 'r' / 'types.sql'
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        # Execute each CREATE SCRIPT statement
        statements = sql_content.split('/')
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and 'CREATE' in stmt.upper():
                self.query(stmt)

    def test_run_func_is_empty(self):
        rows = self.query('''
            SELECT
                fn1.run_func_is_empty() IS NULL
            FROM DUAL''')
        self.assertRowsEqual([(True,)], rows)


class BottleneckTest(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Load R scripts from SQL file
        sql_file = pathlib.Path(__file__).parent.parent.parent.parent / 'lang' / 'r' / 'types.sql'
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        # Execute each CREATE SCRIPT statement
        statements = sql_content.split('/')
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and 'CREATE' in stmt.upper():
                self.query(stmt)

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
