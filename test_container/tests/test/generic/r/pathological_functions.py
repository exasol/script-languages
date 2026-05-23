#!/usr/bin/env python3

from exasol_python_test_framework import udf
import pathlib
import re


class Test(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Load R scripts from SQL file
        sql_file = pathlib.Path(__file__).parent.parent.parent.parent / 'lang' / 'r' / 'pathological_functions.sql'
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        # Execute each CREATE SCRIPT statement
        statements = re.split(r'^\s*/\s*$', sql_content, flags=re.MULTILINE)
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and 'CREATE' in stmt.upper():
                self.query(stmt)


    def test_query_timeout(self):
        self.query('ALTER SESSION SET QUERY_TIMEOUT = 10')
        try:
            with self.assertRaisesRegex(Exception, 'Successfully reconnected after query timeout'):
                self.query('SELECT fn1.sleep(100) FROM dual')
        finally:
            self.query('ALTER SESSION SET QUERY_TIMEOUT = 0')


if __name__ == '__main__':
    udf.main()
