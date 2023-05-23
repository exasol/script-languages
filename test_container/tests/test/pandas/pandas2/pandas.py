#!/usr/bin/env python3

from decimal import Decimal
from datetime import date
from datetime import datetime

from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from typing import List, Tuple, Union

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


    int_dataframe_value_str = "[[1, 2],[3, 4]]"
    int_expected_rows = [(1, 2, None),(3, 4, None)]

    float16_dataframe_value_str = 'np.array([[1.0, 2.0],[3.0, 4.0]], dtype="float16")'
    float_dataframe_value_str = "[[1.0, 2.0],[3.0, 4.0]]"
    float_expected_rows = [(1.0, 2.0, None),(3.0, 4.0, None)]

    str_dataframe_value_str = "[['a','b'],['c','d']]"
    str_expected_rows = [('a','b',None),('c','d',None)]

    bool_dataframe_value_str = "[[True,False],[True,False]]"
    bool_expected_rows = [(True,False,None),(True,False,None)]

    decimal_dataframe_value_str = "[[Decimal('1.0'),Decimal('2.0')],[Decimal('3.0'),Decimal('4.0')]]"
    decimal_expected_rows = [(Decimal('1.0'),Decimal('2.0'),None),(Decimal('3.0'),Decimal('4.0'),None)]

    timestamp_dataframe_value_str = '[[pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251)),' \
                                    +'pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251))],' \
                                    +'[pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251)),' \
                                    +'pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251))]]'
    datetime_dataframe_value_str = '[[datetime(2020, 7, 27, 14, 22, 33, 673251),' \
                                   +'datetime(2020, 7, 27, 14, 22, 33, 673251)],' \
                                   +'[datetime(2020, 7, 27, 14, 22, 33, 673251),' \
                                   +'datetime(2020, 7, 27, 14, 22, 33, 673251)]]'
    datetime_expected_row = [(datetime(2020, 7, 27, 14, 22, 33, 673000),datetime(2020, 7, 27, 14, 22, 33, 673000),None),
                             (datetime(2020, 7, 27, 14, 22, 33, 673000),datetime(2020, 7, 27, 14, 22, 33, 673000),None)]

    none_dataframe_value_str = "[[None, None],[None, None]]"
    none_expected_rows = [(None, None, None),(None, None, None)]
    none_expected_rows_bool_ = [(False, False, None),(False, False, None)]

    nan_dataframe_value_str = "[[np.nan, np.nan],[np.nan, np.nan]]"
    nan_float16_dataframe_value_str = "np.array([[np.nan, np.nan],[np.nan, np.nan]], dtype='float16')"
    nan_expected_rows = [(None, None, None),(None, None, None)]

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
            ("string[pyarrow]", "VARCHAR(2000000)", str_dataframe_value_str, str_expected_rows),
            ("bool[pyarrow]", "boolean", bool_dataframe_value_str, bool_expected_rows),
 
            ("uint8[pyarrow]", "integer", none_dataframe_value_str, none_expected_rows),
            ("uint16[pyarrow]", "integer", none_dataframe_value_str, none_expected_rows),
            ("uint32[pyarrow]", "integer", none_dataframe_value_str, none_expected_rows),
            ("uint64[pyarrow]", "integer", none_dataframe_value_str, none_expected_rows),
            ("int8[pyarrow]", "integer", none_dataframe_value_str, none_expected_rows),
            ("int16[pyarrow]", "integer", none_dataframe_value_str, none_expected_rows),
            ("int32[pyarrow]", "integer", none_dataframe_value_str, none_expected_rows),
            ("int64[pyarrow]", "integer", none_dataframe_value_str, none_expected_rows),
            ("float16[pyarrow]", "float", none_dataframe_value_str, none_expected_rows),
            ("float32[pyarrow]", "float", none_dataframe_value_str, none_expected_rows),
            ("float64[pyarrow]", "float", none_dataframe_value_str, none_expected_rows),
            ("halffloat[pyarrow]", "float", none_dataframe_value_str, none_expected_rows),
            ("float[pyarrow]", "float", none_dataframe_value_str, none_expected_rows),
            ("double[pyarrow]", "float", none_dataframe_value_str, none_expected_rows),
            ("string[pyarrow]", "VARCHAR(2000000)", none_dataframe_value_str, none_expected_rows),
            ("bool[pyarrow]", "boolean", none_dataframe_value_str, none_expected_rows),

            ("uint8[pyarrow]", "integer", nan_dataframe_value_str, nan_expected_rows),
            ("uint16[pyarrow]", "integer", nan_dataframe_value_str, nan_expected_rows),
            ("uint32[pyarrow]", "integer", nan_dataframe_value_str, nan_expected_rows),
            ("uint64[pyarrow]", "integer", nan_dataframe_value_str, nan_expected_rows),
            ("int8[pyarrow]", "integer", nan_dataframe_value_str, nan_expected_rows),
            ("int16[pyarrow]", "integer", nan_dataframe_value_str, nan_expected_rows),
            ("int32[pyarrow]", "integer", nan_dataframe_value_str, nan_expected_rows),
            ("int64[pyarrow]", "integer", nan_dataframe_value_str, nan_expected_rows),
            ("float16[pyarrow]", "float", nan_dataframe_value_str, ".*pyarrow.lib.ArrowNotImplementedError: Unsupported cast from double to halffloat using function cast_half_float.*"),
            ("float32[pyarrow]", "float", nan_dataframe_value_str, nan_expected_rows),
            ("float64[pyarrow]", "float", nan_dataframe_value_str, nan_expected_rows),
            ("halffloat[pyarrow]", "float", nan_dataframe_value_str, ".*pyarrow.lib.ArrowNotImplementedError: Unsupported cast from double to halffloat using function cast_half_float.*"),
            ("float[pyarrow]", "float", nan_dataframe_value_str, nan_expected_rows),
            ("double[pyarrow]", "float", nan_dataframe_value_str, nan_expected_rows),
            ("string[pyarrow]", "VARCHAR(2000000)", nan_dataframe_value_str, nan_expected_rows),
            ("bool[pyarrow]", "boolean", nan_dataframe_value_str, nan_expected_rows),
        ]

    @useData(types)
    def test_pyarrow_dtype_emit(self, dtype:str, sql_type:str, dataframe_value_str:str, expected_result:Union[str,List[Tuple]]):
        sql=udf.fixindent(f'''
            CREATE OR REPLACE PYTHON3 SET SCRIPT test_dtype_emit(i integer) 
            EMITS (o1 {sql_type}, o2 {sql_type}, traceback varchar(2000000)) AS

            def run(ctx):
                try:
                    from decimal import Decimal
                    import pandas as pd
                    import numpy as np
                    from datetime import datetime
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
            rows = self.query('''SELECT test_dtype_emit(0)''')
            if isinstance(expected_result,str):
                self.assertRegex(rows[0][2], expected_result)
            else:
                self.assertRowsEqual(expected_result, rows)

if __name__ == '__main__':
    udf.main()
