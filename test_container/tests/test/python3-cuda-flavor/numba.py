#!/usr/bin/env python3

from exasol_python_test_framework import udf


class NumbaTest(udf.TestCase):
    def setUp(self):
        self.query('create schema numbabasic', ignore_errors=True)

    def test_numba(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
                test_numba(epochs INTEGER)
                RETURNS VARCHAR(10000) AS
                
                import math
                from numba import vectorize, cuda
                import numpy as np
                import os
                
                @vectorize(['float32(float32, float32, float32)',
                            'float64(float64, float64, float64)',],
                             #target='cuda'
                            )
                def cu_discriminant(a, b, c):
                    return math.sqrt(b ** 2 - 4 * a * c)
                
                def run(ctx):
                    N = ctx.epochs
                    dtype = np.float32
                
                    # prepare the input
                    A = np.array(np.random.sample(N), dtype=dtype)
                    B = np.array(np.random.sample(N) + 10, dtype=dtype)
                    C = np.array(np.random.sample(N), dtype=dtype)
                
                    D = cu_discriminant(A, B, C)
                
                    return str(D)
                /
                '''))

        row = self.query("SELECT numbabasic.test_numba(10000);")[0]


if __name__ == '__main__':
    udf.main()
