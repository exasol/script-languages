#!/usr/bin/env python3

import sys

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import useData


class _Python3UdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT basic_range(n INTEGER)
            EMITS (n INTEGER) AS
            def run(ctx):
                if ctx.n is not None:
                    for i in range(ctx.n):
                        ctx.emit(i)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT vectorsize(length INT, dummy DOUBLE)
            RETURNS VARCHAR(2000000) AS
            import gc
            import sys
            
            if sys.version_info[0] == 3: xrange = range
            
            cache = {}
            cache_size = 0
            cache_max = 1024*1024*64
            
            def run(ctx):
                global cache_size, cache
                if ctx.length not in cache:
                    curstr = ''.join([str(i) for i in xrange(ctx.length)])
                    if cache_size + len(curstr) > cache_max:
                        cache = {}
                        cache_size = 0
                        gc.collect()
                    cache[ctx.length] = curstr
                    cache_size += len(curstr)
                return cache[ctx.length]
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT vectorsize5000(A DOUBLE)
            RETURNS VARCHAR(2000000) AS
            retval = ''.join([str(i) for i in range(5000)])
            
            def run(ctx):
                return retval
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT vectorsize_set(length INT, n INT, dummy DOUBLE)
            EMITS (o VARCHAR(2000000)) AS
            import gc
            import sys
            
            if sys.version_info[0] == 3: xrange = range
            
            cache = {}
            cache_size = 0
            cache_max = 1024*1024*64
            
            def run(ctx):
                global cache_size, cache
                if ctx.length not in cache:
                    curstr = ''.join([str(i) for i in xrange(ctx.length)])
                    if cache_size + len(curstr) > cache_max:
                        cache = {}
                        cache_size = 0
                        gc.collect()
                    cache[ctx.length] = curstr
                    cache_size += len(curstr)
                for i in xrange(ctx.n):
                    ctx.emit(cache[ctx.length])
            /
        '''))

class Vectorsize(_Python3UdfSetup):

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
            raise udf.SkipTest('test is to slow')

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
