#!/usr/bin/env python3

import sys

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import requires, useData, SkipTest


class Vectorsize(udf.TestCase):

    @requires('VECTORSIZE5000')
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
    @requires('VECTORSIZE')
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
    @requires('VECTORSIZE_SET')
    @requires('BASIC_RANGE')
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
