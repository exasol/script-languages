#!/usr/bin/env python3

import sys

from exasol_python_test_framework import udf
import pathlib
from exasol_python_test_framework.udf import useData, SkipTest


class Vectorsize(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        
        # Load R scripts from SQL file
        sql_file = pathlib.Path(__file__).parent.parent.parent.parent / 'lang' / 'r' / 'vectorsize.sql'
        with open(sql_file, 'r') as f:
            sql_content = f.read()
        
        # Execute each CREATE SCRIPT statement
        statements = sql_content.split('/')
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and 'CREATE' in stmt.upper():
                self.query(stmt)


    def test_vectorsize_5000(self):
        self.query('''
		        SELECT max(fn1.vectorsize5000(float1))
                FROM TEST.ENGINETABLEBIG1''')

    data = [
            (10,),
            (30,),
            (100,),
            (300,),
            (1000,),
            (3000,),
            (10000,),
            (30000,),
            (100000,),
            (200000,),
            (351850,),
            ]

    @useData(data)
    def test_vectorsize(self, size):
        limits = {
            'lua':      100000,
            'python3':   8000,
            'r':        3000,
            'java':     3000
            }
        if size > limits.get(udf.opts.lang, sys.maxsize):
            raise SkipTest('test is to slow')

        self.query('''
                SELECT max(fn1.vectorsize(%d, float1))
                FROM TEST.ENGINETABLEBIG1
                ''' % size)

    data = [
            (10, 10, 10),
            (100, 100, 100),
            (1000, 100, 100),
            (10000, 100, 100),
            (100000, 100, 100),
            (351850, 100, 100),
            (100, 10, 100000),
            (100, 100, 10000),
            (100, 1000, 1000),
            (100, 10000, 100),
            (100, 100000, 10),
            ]

    @useData(data)
    def test_vectorsize_set(self, a, b, c):
        q = '''
                SELECT max(o)
                FROM (
                    SELECT fn1.vectorsize_set(%d, %d, n)
                    FROM (
                        SELECT fn1.basic_range(%d)
                        FROM DUAL
                    )
                )
                ''' % (a, b, c)
        self.query(q)


if __name__ == '__main__':
    udf.main()
