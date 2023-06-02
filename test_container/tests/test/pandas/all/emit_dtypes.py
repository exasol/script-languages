#!/usr/bin/env python3

from decimal import Decimal
from datetime import date
from datetime import datetime

from  exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from typing import List, Tuple, Union


class PandasDataFrameEmitDTypes(udf.TestCase):
    def setUp(self):
        self.maxDiff=None

        self.query(f'CREATE SCHEMA {self.__class__.__name__}', ignore_errors=True)
        self.query(f'OPEN SCHEMA {self.__class__.__name__}', ignore_errors=True)

    int_dataframe_value_str = "[[1, 2],[3, 4]]"
    int_expected_rows = [(1, 2, None),(3, 4, None)]
    int_to_float_expected_rows = [(1.0, 2.0, None),(3.0, 4.0, None)]

    float16_dataframe_value_str = 'np.array([[1.1, 2.1],[3.1, 4.1]], dtype="float16")'
    float_dataframe_value_str = "[[1.1, 2.1],[3.1, 4.1]]"
    float_expected_rows = [(1.1, 2.1, None),(3.1, 4.1, None)]

    str_dataframe_value_str = "[['a','b'],['c','d']]"
    str_expected_rows = [('a','b',None),('c','d',None)]

    bool_dataframe_value_str = "[[True,False],[True,False]]"
    bool_expected_rows = [(True,False,None),(True,False,None)]

    decimal_dataframe_value_str = "[[Decimal('1.1'),Decimal('2.1')],[Decimal('3.1'),Decimal('4.1')]]"
    decimal_expected_rows = [(Decimal('1.1'),Decimal('2.1'),None),(Decimal('3.1'),Decimal('4.1'),None)]
    int_to_decimal_expected_rows = [(Decimal('1'),Decimal('2'),None),(Decimal('3'),Decimal('4'),None)]

    timestamp_dataframe_value_str = '[[pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251)),' \
                                    +'pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251))],' \
                                    +'[pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251)),' \
                                    +'pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251))]]'
    datetime_dataframe_value_str = '[[datetime(2020, 7, 27, 14, 22, 33, 673251),' \
                                   +'datetime(2020, 7, 27, 14, 22, 33, 673251)],' \
                                   +'[datetime(2020, 7, 27, 14, 22, 33, 673251),' \
                                   +'datetime(2020, 7, 27, 14, 22, 33, 673251)]]'
    datetime_expected_rows = [(datetime(2020, 7, 27, 14, 22, 33, 673000),datetime(2020, 7, 27, 14, 22, 33, 673000),None),
                             (datetime(2020, 7, 27, 14, 22, 33, 673000),datetime(2020, 7, 27, 14, 22, 33, 673000),None)]
    date_dataframe_value_str = '[[date(2020, 7, 27),' \
                                   +'date(2020, 7, 27)],' \
                                   +'[date(2020, 7, 27),' \
                                   +'date(2020, 7, 27)]]'
    date_expected_rows = [(date(2020, 7, 27),date(2020, 7, 27),None),
                         (date(2020, 7, 27),date(2020, 7, 27),None)]

    mixed_int_dataframe_value_str = "[[1, None],[None, 4]]"
    mixed_int_expected_rows = [(1, None, None),(None, 4, None)]
    mixed_int_to_float_expected_rows = [(1.0, None, None),(None, 4.0, None)]

    mixed_float16_dataframe_value_str = 'np.array([[1.1, None],[None, 4.1]], dtype="float16")'
    mixed_float_dataframe_value_str = "[[1.1, None],[None, 4.1]]"
    mixed_float_expected_rows = [(1.1, None, None),(None, 4.1, None)]

    mixed_str_dataframe_value_str = "[['a',None],[None,'d']]"
    mixed_str_expected_rows = [('a',None,None),(None,'d',None)]

    mixed_bool_dataframe_value_str = "[[True,None],[None,False]]"
    mixed_bool_expected_rows = [(True,None,None),(None,False,None)]
    mixed_bool_expected_rows_bool_ = [(True, False, None),(False, False, None)]

    mixed_decimal_dataframe_value_str = "[[Decimal('1.1'),None],[None,Decimal('4.1')]]"
    mixed_decimal_expected_rows = [(Decimal('1.1'),None,None),(None,Decimal('4.1'),None)]
    mixed_int_to_decimal_expected_rows = [(Decimal('1'),None,None),(None,Decimal('4'),None)]

    mixed_timestamp_dataframe_value_str = '[[pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251)),None],' \
                                    +'[None,pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251))]]'
    mixed_datetime_dataframe_value_str = '[[datetime(2020, 7, 27, 14, 22, 33, 673251),None],' \
                                   +'[None,datetime(2020, 7, 27, 14, 22, 33, 673251)]]'
    mixed_datetime_expected_rows = [(datetime(2020, 7, 27, 14, 22, 33, 673000),None,None),
                                    (None,datetime(2020, 7, 27, 14, 22, 33, 673000),None)]
    mixed_date_dataframe_value_str = '[[date(2020, 7, 27),None],' \
                                   +'[None,date(2020, 7, 27)]]'
    mixed_date_expected_rows = [(date(2020, 7, 27),None,None),
                                (None,date(2020, 7, 27),None)]

    none_dataframe_value_str = "[[None, None],[None, None]]"
    none_expected_rows = [(None, None, None),(None, None, None)]
    none_expected_rows_bool_ = [(False, False, None),(False, False, None)]

    nan_dataframe_value_str = "[[np.nan, np.nan],[np.nan, np.nan]]"
    nan_expected_rows = [(None, None, None),(None, None, None)]
    nan_expected_rows_bool_ = [(True, True, None),(True, True, None)]

    

    types = [
            # Full columns without None or NaN / Int

            ("uint8", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("uint16", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("uint32", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("uint64", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("int8", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("int16", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("int32", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("int64", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("object", "integer", int_dataframe_value_str, int_expected_rows, False),

            # Full columns without None or NaN / Float

            ("float16", "double", float16_dataframe_value_str, float_expected_rows, True),
            ("float32", "double", float_dataframe_value_str, float_expected_rows, True),
            ("float64", "double", float_dataframe_value_str, float_expected_rows, False),
            ("float", "double", float_dataframe_value_str, float_expected_rows, False),
            ("double", "double", float_dataframe_value_str, float_expected_rows, False),
            ("object", "double", float_dataframe_value_str, float_expected_rows, False),

            # Full columns without None or NaN / Int to Float

            ("uint8", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("uint16", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("uint32", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("uint64", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("int8", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("int16", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("int32", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("int64", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("object", "double", int_dataframe_value_str, int_to_float_expected_rows, False),

            # Full columns without None or NaN / Float to Int

            ("float16", "integer", float16_dataframe_value_str, int_expected_rows, False),
            ("float32", "integer", float_dataframe_value_str, int_expected_rows, False),
            ("float64", "integer", float_dataframe_value_str, int_expected_rows, False),
            ("float", "integer", float_dataframe_value_str, int_expected_rows, False),
            ("double", "integer", float_dataframe_value_str, int_expected_rows, False),
            ("object", "integer", float_dataframe_value_str, int_expected_rows, False),

            # Full columns without None or NaN / Int to Decimal

            ("uint8", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("uint16", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("uint32", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("uint64", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("int8", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("int16", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("int32", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("int64", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("object", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),

            # Full columns without None or NaN / Float to Decimal

            ("float16", "DECIMAL(10,5)", float16_dataframe_value_str, decimal_expected_rows, True),
            ("float32", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, True),
            ("float64", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, True),
            ("float", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, True),
            ("double", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, True),
            ("object", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, True),
 
            # Full columns without None or NaN / Decimal

            ("object", "DECIMAL(10,5)", decimal_dataframe_value_str, decimal_expected_rows, False),

            # Full columns without None or NaN / String

            ("string", "VARCHAR(2000000)", str_dataframe_value_str, str_expected_rows, False),
            ("object", "VARCHAR(2000000)", str_dataframe_value_str, str_expected_rows, False),

            # Full columns without None or NaN / Boolean

            ("bool_", "boolean", bool_dataframe_value_str, bool_expected_rows, False),
            ("boolean", "boolean", bool_dataframe_value_str, bool_expected_rows, False),
            ("object", "boolean", bool_dataframe_value_str, bool_expected_rows, False),

            # Full columns without None or NaN / Date and Time

            ("datetime64[ns]", "timestamp", timestamp_dataframe_value_str, datetime_expected_rows, False),
            ("object", "timestamp", timestamp_dataframe_value_str, datetime_expected_rows, False),
            ("object", "timestamp", datetime_dataframe_value_str, ".*F-UDF-CL-SL-PYTHON-1056.*unexpected python type: py_datetime.datetime.*", False),
            ("object", "timestamp", date_dataframe_value_str, ".*F-UDF-CL-SL-PYTHON-1071: emit column 0 of type TIMESTAMP but data given have type py_datetime.date.*", False),
            ("object", "DATE", date_dataframe_value_str, date_expected_rows, False),

            # Mixed columns with values and None / Int

            #(u)int-dtypes don't support None or np.nan

            ("object", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),

            # Mixed columns with values and None / Float

            ("float16", "double", mixed_float16_dataframe_value_str, mixed_float_expected_rows, True),
            ("float32", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, True),
            ("float64", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, False),
            ("float", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, False),
            ("double", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, False),
            ("object", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, False),

            # Mixed columns with values and None / Float to Int
            ("float16", "integer", mixed_float16_dataframe_value_str, mixed_int_expected_rows, False),
            ("float32", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),
            ("float64", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),
            ("float", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),
            ("double", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),
            ("object", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),

            # Mixed columns with values and None / Int to Decimal

            ("object", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),

            # Mixed columns with values and None / Float to Decimal

            ("float16", "DECIMAL(10,5)", mixed_float16_dataframe_value_str, mixed_decimal_expected_rows, True),
            ("float32", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, True),
            ("float64", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, True),
            ("float", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, True),
            ("double", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, True),
            ("object", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, True),

            # Mixed columns with values and None / Decimal

            ("object", "DECIMAL(10,5)", mixed_decimal_dataframe_value_str, mixed_decimal_expected_rows, False),

            # Mixed columns with values and None / String

            ("string", "VARCHAR(2000000)", mixed_str_dataframe_value_str, mixed_str_expected_rows, False),
            ("object", "VARCHAR(2000000)", mixed_str_dataframe_value_str, mixed_str_expected_rows, False),
 
            # Mixed columns with values and None / Boolean

            ("bool_", "boolean", mixed_bool_dataframe_value_str, mixed_bool_expected_rows_bool_, False),
            ("boolean", "boolean", mixed_bool_dataframe_value_str, mixed_bool_expected_rows, False),
            ("object", "boolean", mixed_bool_dataframe_value_str, mixed_bool_expected_rows, False),

            # Mixed columns with values and None / Data and time

            ("datetime64[ns]", "timestamp", mixed_timestamp_dataframe_value_str, mixed_datetime_expected_rows, False),
            ("object", "timestamp", mixed_timestamp_dataframe_value_str, mixed_datetime_expected_rows, False),
            ("object", "DATE", mixed_date_dataframe_value_str, mixed_date_expected_rows, False),

            # None

            ("object", "integer", none_dataframe_value_str, none_expected_rows, False),

            ("float16", "double", none_dataframe_value_str, none_expected_rows, False),
            ("float32", "double", none_dataframe_value_str, none_expected_rows, False),
            ("float64", "double", none_dataframe_value_str, none_expected_rows, False),
            ("float", "double", none_dataframe_value_str, none_expected_rows, False),
            ("double", "double", none_dataframe_value_str, none_expected_rows, False),
            ("object", "double", none_dataframe_value_str, none_expected_rows, False),

            ("float16", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("float32", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("float64", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("float", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("double", "integer", none_dataframe_value_str, none_expected_rows, False),

            ("float16", "DECIMAL(10,5)", none_dataframe_value_str, none_expected_rows, False),
            ("float32", "DECIMAL(10,5)", none_dataframe_value_str, none_expected_rows, False),
            ("float64", "DECIMAL(10,5)", none_dataframe_value_str, none_expected_rows, False),
            ("float", "DECIMAL(10,5)", none_dataframe_value_str, none_expected_rows, False),
            ("double", "DECIMAL(10,5)", none_dataframe_value_str, none_expected_rows, False),

            ("object", "DECIMAL(10,5)", none_dataframe_value_str, none_expected_rows, False),

            ("string", "VARCHAR(2000000)", none_dataframe_value_str, none_expected_rows, False),
            ("object", "VARCHAR(2000000)", none_dataframe_value_str, none_expected_rows, False),

            ("bool_", "boolean", none_dataframe_value_str, none_expected_rows_bool_, False),
            ("boolean", "boolean", none_dataframe_value_str, none_expected_rows, False),
            ("object", "boolean", none_dataframe_value_str, none_expected_rows, False),

            ("datetime64[ns]", "timestamp", none_dataframe_value_str, none_expected_rows, False),
            ("object", "timestamp", none_dataframe_value_str, none_expected_rows, False),
            ("object", "DATE", none_dataframe_value_str, none_expected_rows, False),

            # NaN

            ("object", "integer", nan_dataframe_value_str, nan_expected_rows, False),

            ("float16", "double", nan_dataframe_value_str, nan_expected_rows, False),
            ("float32", "double", nan_dataframe_value_str, nan_expected_rows, False),
            ("float64", "double", nan_dataframe_value_str, nan_expected_rows, False),
            ("float", "double", nan_dataframe_value_str, nan_expected_rows, False),
            ("double", "double", nan_dataframe_value_str, nan_expected_rows, False),
            ("object", "double", nan_dataframe_value_str, nan_expected_rows, False),

            ("float16", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("float32", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("float64", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("float", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("double", "integer", nan_dataframe_value_str, nan_expected_rows, False),

            ("float16", "DECIMAL(10,5)", nan_dataframe_value_str, nan_expected_rows, False),
            ("float32", "DECIMAL(10,5)", nan_dataframe_value_str, nan_expected_rows, False),
            ("float64", "DECIMAL(10,5)", nan_dataframe_value_str, nan_expected_rows, False),
            ("float", "DECIMAL(10,5)", nan_dataframe_value_str, nan_expected_rows, False),
            ("double", "DECIMAL(10,5)", nan_dataframe_value_str, nan_expected_rows, False),

            #("object", "DECIMAL(10,5)", nan_dataframe_value_str, None, False), # Fails with VM error: [22018] invalid character value for cast; Value: 'nan'

            ("string", "VARCHAR(2000000)", nan_dataframe_value_str, nan_expected_rows, False),
            ("object", "VARCHAR(2000000)", nan_dataframe_value_str, ".*PYTHON-1140: emit column 0 of type STRING but data given have type py_float.*", False),

            ("bool_", "boolean", nan_dataframe_value_str, nan_expected_rows_bool_, False),
            ("boolean", "boolean", nan_dataframe_value_str, nan_expected_rows, False),
            ("object", "boolean", nan_dataframe_value_str, ".*F-UDF-CL-SL-PYTHON-1140: emit column 0 of type BOOLEAN but data given have type py_float.*", False),

            ("datetime64[ns]", "timestamp", nan_dataframe_value_str, nan_expected_rows, False),
            ("object", "timestamp", nan_dataframe_value_str, ".*F-UDF-CL-SL-PYTHON-1140: emit column 0 of type TIMESTAMP but data given have type py_float.*", False),
            ("object", "DATE", nan_dataframe_value_str, ".*F-UDF-CL-SL-PYTHON-1140: emit column 0 of type DATE but data given have type py_float.*", False),

        ]

    @useData(types)
    def test_dtype_emit(self, dtype:str, sql_type:str, dataframe_value_str:str, expected_result:Union[str,List[Tuple]], use_almost_equal:bool):
        sql=udf.fixindent(f'''
            CREATE OR REPLACE PYTHON3 SET SCRIPT test_dtype_emit(i integer) 
            EMITS (o1 {sql_type}, o2 {sql_type}, traceback varchar(2000000)) AS

            def run(ctx):
                try:
                    from decimal import Decimal
                    import pandas as pd
                    import numpy as np
                    from datetime import datetime, date
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
                if use_almost_equal:
                    self.assertRowsAlmostEqual(expected_result, rows, places=1)
                else:
                    self.assertRowsEqual(expected_result, rows)

    def isValueAlmostEqual(self, left, right, places):
        if isinstance(left, (float, Decimal)) and isinstance(right, (float, Decimal)):
            return round(left, places) == round(right, places)
        else:
            return left == right

    def isRowAlmostEqual(self, left, right, places):
        if len(left) != len(right):
            return False
        all_values_almost_equal = all(self.isValueAlmostEqual(lvalue, rvalue, places)
                                      for lvalue, rvalue in zip(left, right))
        return all_values_almost_equal
 
    def assertRowsAlmostEqual(self, left, right, places):
        lrows = [tuple(x) for x in left]
        rrows = [tuple(x) for x in right]
        if len(lrows) != len(rrows):
            raise AssertionError(f'{lrows} and {rrows} have different number of rows.')
        all_rows_almost_equal = all(self.isRowAlmostEqual(lrow, rrow, places) for lrow, rrow in zip(lrows, rrows))
        if not all_rows_almost_equal:
             raise AssertionError(f'{lrows} and {rrows} are not almost equal.')

if __name__ == '__main__':
    udf.main()

