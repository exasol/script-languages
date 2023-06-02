#!/usr/bin/env python3


from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.udf_debug import UdfDebugger

class NumbaTest(udf.TestCase):
    def setUp(self): 
        self.query('create schema numbatest', ignore_errors=True)

    def test_numba(self):
        sql=udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT numbatest.test_numba(i integer) EMITS (o DOUBLE) AS
            from numba import jit
            import numpy as np
            
            @jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
            def go_fast(a): # Function is compiled to machine code when called the first time
                trace = 0.0
                for i in range(a.shape[0]):   # Numba likes loops
                    trace += np.tanh(a[i, i]) # Numba likes NumPy functions
                return np.sum(a + trace)      # Numba likes NumPy broadcasting

            def run(ctx):
                while True:
                    x = np.arange(100).reshape(10, 10) * ctx.i
                    res = go_fast(x)
                    ctx.emit(res)
                    if not ctx.next():
                        break
            /
            ''')
        print(sql)
        self.query(sql)
        with UdfDebugger(test_case=self):
            rows = self.query('''SELECT numbatest.test_numba(t.i) FROM values between 1 and 100 as t(i)''')
            self.assertEqual(100, len(rows))


if __name__ == '__main__':
    udf.main()
