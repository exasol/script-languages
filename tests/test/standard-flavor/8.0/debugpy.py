#!/usr/bin/env python3


from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.udf_debug import UdfDebugger

class DebugPyTest(udf.TestCase):
    def setUp(self): 
        self.query('create schema debugpytest', ignore_errors=True)

    def test_debugpy(self):
        sql=udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT debugpytest.test_debugpy() returns int AS
            import debugpy
            debugpy.listen(("0.0.0.0", 5678))
            
            def run(ctx): return 1
            /
            ''')
        print(sql)
        self.query(sql)
        with UdfDebugger(test_case=self):
            rows = self.query('''SELECT debugpytest.test_debugpy() FROM dual''')
            self.assertRowsEqual([(1,)], rows)


if __name__ == '__main__':
    udf.main()
