#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import requires, useData, SkipTest

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
            'ext-python':   8000,
            'python':   8000,
            'r':        3000,
            'java':     3000
            }
        if size > limits.get(udf.opts.lang, sys.maxint):
            raise SkipTest('test is to slow')

        self.query('''
                SELECT max(fn1.vectorsize(%d, float1))
                FROM TEST.ENGINETABLEBIG1
                ''' % size
                ) 


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
        self.query('''
                SELECT max(o)
                FROM (
                    SELECT fn1.vectorsize_set(%d, %d, n)
                    FROM (
                        SELECT fn1.basic_range(%d)
                        FROM DUAL
                    )
                )
                ''' % (a, b, c)
                )

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
