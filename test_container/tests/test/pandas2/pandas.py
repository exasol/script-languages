#!/usr/bin/env python3


from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from typing import List, Tuple

class Pandas2Test(udf.TestCase):
    def setUp(self): 
        self.query('create schema pandas2test', ignore_errors=True)
        self.maxDiff=None

    def test_pandas2_version(self):
        sql=udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT pandas2test.test_pandas2_version(i integer) EMITS (o VARCHAR(100)) AS

            def run(ctx):
                import pandas as pd
                ctx.emit(pd.__version__)
            /
            ''')
        print(sql)
        self.query(sql)
        rows = self.query('''SELECT pandas2test.test_pandas2_version(0)''')
        version_parts = rows[0][0].split(".")
        self.assertEqual("2",version_parts[0])

    int_dataframe_value_str = "[[1,2],[3,4]]"
    int_expected_rows = [(1,2, None),(3,4, None)]
    float16_dataframe_value_str = 'np.array([[1.0,1.0],[1.0,1.0]], dtype="float16")'
    float_dataframe_value_str = "[[1.0,1.0],[1.0,1.0]]"
    float_expected_rows = [(1.0,1.0, None),(1.0,1.0, None)]


    types = [
            ("uint8[pyarrow]", "integer", int_dataframe_value_str, int_expected_rows),
            ("uint16[pyarrow]", "integer", int_dataframe_value_str, int_expected_rows),
            ("uint32[pyarrow]", "integer", int_dataframe_value_str, int_expected_rows),
            ("uint64[pyarrow]", "integer", int_dataframe_value_str, int_expected_rows),
            ("int8[pyarrow]", "integer", int_dataframe_value_str, int_expected_rows),
            ("int16[pyarrow]", "integer", int_dataframe_value_str, int_expected_rows),
            ("int32[pyarrow]", "integer", int_dataframe_value_str, int_expected_rows),
            ("int64[pyarrow]", "integer", int_dataframe_value_str, int_expected_rows),
            ("float16[pyarrow]", "float", float16_dataframe_value_str, float_expected_rows),
            ("float32[pyarrow]", "float", float_dataframe_value_str, float_expected_rows),
            ("float64[pyarrow]", "float", float_dataframe_value_str, float_expected_rows),
            ("halffloat[pyarrow]", "float", float16_dataframe_value_str, float_expected_rows),
            ("float[pyarrow]", "float", float_dataframe_value_str, float_expected_rows),
            ("double[pyarrow]", "float", float_dataframe_value_str, float_expected_rows),
        ]

    @useData(types)
    def test_pandas2_int_pyarrow_dtype_emit(self, dtype:str, sql_type:str, dataframe_value_str:str, expected_rows:List[Tuple]):
        sql=udf.fixindent(f'''
            CREATE OR REPLACE PYTHON3 SET SCRIPT pandas2test.test_pandas2_pyarrow_dtype_emit(i integer) 
            EMITS (o1 {sql_type}, o2 {sql_type}, traceback varchar(2000000)) AS

            def run(ctx):
                try:
                    import pandas as pd
                    import numpy as np
                    df = pd.DataFrame({dataframe_value_str}, dtype="{dtype}")
                    df["traceback"]=None
                    ctx.emit(df)
                except:
                    import traceback
                    ctx.emit(None,None,traceback.format_exc())
            /
            ''')
        print(sql)
        self.query(sql)
        with UdfDebugger(test_case=self):
            rows = self.query('''SELECT pandas2test.test_pandas2_pyarrow_dtype_emit(0)''')
            self.assertRowsEqual(expected_rows, rows)


if __name__ == '__main__':
    udf.main()
