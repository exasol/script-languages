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

    nan_dataframe_value_str = "[[np.nan, np.nan],[np.nan, np.nan]]"
    nan_expected_rows = [(None, None, None),(None, None, None)]

    types = [
            # Full columns without None or NaN / Int

            ("dtype='uint8[pyarrow]'", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("dtype='uint16[pyarrow]'", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("dtype='uint32[pyarrow]'", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("dtype='uint64[pyarrow]'", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("dtype='int8[pyarrow]'", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("dtype='int16[pyarrow]'", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("dtype='int32[pyarrow]'", "integer", int_dataframe_value_str, int_expected_rows, False),
            ("dtype='int64[pyarrow]'", "integer", int_dataframe_value_str, int_expected_rows, False),

            # Full columns without None or NaN / Float

            ("dtype='float16[pyarrow]'", "double", float16_dataframe_value_str, float_expected_rows, True),
            ("dtype='float32[pyarrow]'", "double", float_dataframe_value_str, float_expected_rows, True),
            ("dtype='float64[pyarrow]'", "double", float_dataframe_value_str, float_expected_rows, False),
            ("dtype='halffloat[pyarrow]'", "double", float16_dataframe_value_str, float_expected_rows, True),
            ("dtype='float[pyarrow]'", "double", float_dataframe_value_str, float_expected_rows, True),
            ("dtype='double[pyarrow]'", "double", float_dataframe_value_str, float_expected_rows, False),           

            # Full columns without None or NaN / Decimal

            ("dtype=pd.ArrowDtype(pa.decimal128(3, scale=2))", "DECIMAL(10,5)", decimal_dataframe_value_str, decimal_expected_rows, False),
            # Full columns without None or NaN / Int to Decimal

            ("dtype='uint8[pyarrow]'", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("dtype='uint16[pyarrow]'", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("dtype='uint32[pyarrow]'", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("dtype='uint64[pyarrow]'", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("dtype='int8[pyarrow]'", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("dtype='int16[pyarrow]'", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("dtype='int32[pyarrow]'", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),
            ("dtype='int64[pyarrow]'", "DECIMAL(10,5)", int_dataframe_value_str, int_to_decimal_expected_rows, False),

            # Full columns without None or NaN / Float to Decimal

            ("dtype='float16[pyarrow]'", "DECIMAL(10,5)", float16_dataframe_value_str, decimal_expected_rows, True),
            ("dtype='float32[pyarrow]'", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, False),
            ("dtype='float64[pyarrow]'", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, False),
            ("dtype='halffloat[pyarrow]'", "DECIMAL(10,5)", float16_dataframe_value_str, decimal_expected_rows, True),
            ("dtype='float[pyarrow]'", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, False),
            ("dtype='double[pyarrow]'", "DECIMAL(10,5)", float_dataframe_value_str, decimal_expected_rows, False),           
            
            # Full columns without None or NaN / Int To Double

            ("dtype='uint8[pyarrow]'", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("dtype='uint16[pyarrow]'", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("dtype='uint32[pyarrow]'", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("dtype='uint64[pyarrow]'", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("dtype='int8[pyarrow]'", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("dtype='int16[pyarrow]'", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("dtype='int32[pyarrow]'", "double", int_dataframe_value_str, int_to_float_expected_rows, False),
            ("dtype='int64[pyarrow]'", "double", int_dataframe_value_str, int_to_float_expected_rows, False),

            # Full columns without None or NaN / Float to Int

            ("dtype='float16[pyarrow]'", "integer", float16_dataframe_value_str, int_expected_rows, False),
            ("dtype='float32[pyarrow]'", "integer", float_dataframe_value_str, int_expected_rows, False),
            ("dtype='float64[pyarrow]'", "integer", float_dataframe_value_str, int_expected_rows, False),
            ("dtype='halffloat[pyarrow]'", "integer", float16_dataframe_value_str, int_expected_rows, False),
            ("dtype='float[pyarrow]'", "integer", float_dataframe_value_str, int_expected_rows, False),
            ("dtype='double[pyarrow]'", "integer", float_dataframe_value_str, int_expected_rows, False),           

            # Full columns without None or NaN / String

            ("dtype='string[pyarrow]'", "VARCHAR(2000000)", str_dataframe_value_str, str_expected_rows, False),

            # Full columns without None or NaN / Boolean

            ("dtype='bool[pyarrow]'", "boolean", bool_dataframe_value_str, bool_expected_rows, False),

            # Full columns without None or NaN / Date and time

            ("dtype=pd.ArrowDtype(pa.timestamp('ns','UTC'))", "timestamp", datetime_dataframe_value_str, datetime_expected_rows, False),
            #df = pd.DataFrame([[datetime.date(2012,1,1),None],[None,None]], dtype=pd.ArrowDtype(pa.date32())) can't be created at the moment, because it fails with "AttributeError: 'ArrowDtype' object has no attribute 'tz'" and pa.date32() doesn't accept a timezone
            #df = pd.DataFrame([[datetime.date(2012,1,1),None],[None,None]], dtype=pd.ArrowDtype(pa.date64())) can't be created at the moment, because it fails with "AttributeError: 'ArrowDtype' object has no attribute 'tz'" and pa.date32() doesn't accept a timezone

            # Mixed columns with values and None / Int

            ("dtype='uint8[pyarrow]'", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='uint16[pyarrow]'", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='uint32[pyarrow]'", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='uint64[pyarrow]'", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='int8[pyarrow]'", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='int16[pyarrow]'", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='int32[pyarrow]'", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='int64[pyarrow]'", "integer", mixed_int_dataframe_value_str, mixed_int_expected_rows, False),

            # Mixed columns with values and None / Float

            ("dtype='float16[pyarrow]'", "double", mixed_float16_dataframe_value_str, mixed_float_expected_rows, True),
            ("dtype='float32[pyarrow]'", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, True),
            ("dtype='float64[pyarrow]'", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, False),
            ("dtype='halffloat[pyarrow]'", "double", mixed_float16_dataframe_value_str, mixed_float_expected_rows, True),
            ("dtype='float[pyarrow]'", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, True),
            ("dtype='double[pyarrow]'", "double", mixed_float_dataframe_value_str, mixed_float_expected_rows, False),           

            # Mixed columns with values and None / Decimal

            ("dtype=pd.ArrowDtype(pa.decimal128(3, scale=2))", "DECIMAL(10,5)", mixed_decimal_dataframe_value_str, mixed_decimal_expected_rows, False),
            # Mixed columns with values and None / Int to Decimal

            ("dtype='uint8[pyarrow]'", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),
            ("dtype='uint16[pyarrow]'", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),
            ("dtype='uint32[pyarrow]'", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),
            ("dtype='uint64[pyarrow]'", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),
            ("dtype='int8[pyarrow]'", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),
            ("dtype='int16[pyarrow]'", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),
            ("dtype='int32[pyarrow]'", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),
            ("dtype='int64[pyarrow]'", "DECIMAL(10,5)", mixed_int_dataframe_value_str, mixed_int_to_decimal_expected_rows, False),

            # Mixed columns with values and None / Float to Decimal

            ("dtype='float16[pyarrow]'", "DECIMAL(10,5)", mixed_float16_dataframe_value_str, mixed_decimal_expected_rows, True),
            ("dtype='float32[pyarrow]'", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, False),
            ("dtype='float64[pyarrow]'", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, False),
            ("dtype='halffloat[pyarrow]'", "DECIMAL(10,5)", mixed_float16_dataframe_value_str, mixed_decimal_expected_rows, True),
            ("dtype='float[pyarrow]'", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, False),
            ("dtype='double[pyarrow]'", "DECIMAL(10,5)", mixed_float_dataframe_value_str, mixed_decimal_expected_rows, False),           
            
            # Mixed columns with values and None / Int To Double

            ("dtype='uint8[pyarrow]'", "double", mixed_int_dataframe_value_str, mixed_int_to_float_expected_rows, False),
            ("dtype='uint16[pyarrow]'", "double", mixed_int_dataframe_value_str, mixed_int_to_float_expected_rows, False),
            ("dtype='uint32[pyarrow]'", "double", mixed_int_dataframe_value_str, mixed_int_to_float_expected_rows, False),
            ("dtype='uint64[pyarrow]'", "double", mixed_int_dataframe_value_str, mixed_int_to_float_expected_rows, False),
            ("dtype='int8[pyarrow]'", "double", mixed_int_dataframe_value_str, mixed_int_to_float_expected_rows, False),
            ("dtype='int16[pyarrow]'", "double", mixed_int_dataframe_value_str, mixed_int_to_float_expected_rows, False),
            ("dtype='int32[pyarrow]'", "double", mixed_int_dataframe_value_str, mixed_int_to_float_expected_rows, False),
            ("dtype='int64[pyarrow]'", "double", mixed_int_dataframe_value_str, mixed_int_to_float_expected_rows, False),

            # Mixed columns with values and None / Float to Int

            ("dtype='float16[pyarrow]'", "integer", mixed_float16_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='float32[pyarrow]'", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='float64[pyarrow]'", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='halffloat[pyarrow]'", "integer", mixed_float16_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='float[pyarrow]'", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),
            ("dtype='double[pyarrow]'", "integer", mixed_float_dataframe_value_str, mixed_int_expected_rows, False),           

            # Mixed columns with values and None / String

            ("dtype='string[pyarrow]'", "VARCHAR(2000000)", mixed_str_dataframe_value_str, mixed_str_expected_rows, False),

            # Mixed columns with values and None / Boolean

            ("dtype='bool[pyarrow]'", "boolean", mixed_bool_dataframe_value_str, mixed_bool_expected_rows, False),

            # Mixed columns with values and None / Date and time

            ("dtype=pd.ArrowDtype(pa.timestamp('ns','UTC'))", "timestamp", mixed_datetime_dataframe_value_str, mixed_datetime_expected_rows, False),

            # None

            ("dtype='uint8[pyarrow]'", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='uint16[pyarrow]'", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='uint32[pyarrow]'", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='uint64[pyarrow]'", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='int8[pyarrow]'", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='int16[pyarrow]'", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='int32[pyarrow]'", "integer", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='int64[pyarrow]'", "integer", none_dataframe_value_str, none_expected_rows, False),

            # Decativated until all flavors are using pyarrow >=16.0.0
            #("dtype='float16[pyarrow]'", "float", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='float32[pyarrow]'", "float", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='float64[pyarrow]'", "float", none_dataframe_value_str, none_expected_rows, False),
            # Decativated until all flavors are using pyarrow >=16.0.0
            #("dtype='halffloat[pyarrow]'", "float", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='float[pyarrow]'", "float", none_dataframe_value_str, none_expected_rows, False),
            ("dtype='double[pyarrow]'", "float", none_dataframe_value_str, none_expected_rows, False),

            ("dtype='string[pyarrow]'", "VARCHAR(2000000)", none_dataframe_value_str, none_expected_rows, False),

            ("dtype='bool[pyarrow]'", "boolean", none_dataframe_value_str, none_expected_rows, False),

            ("dtype=pd.ArrowDtype(pa.timestamp('ns','UTC'))", "timestamp", none_dataframe_value_str, none_expected_rows, False),
            ("dtype=pd.ArrowDtype(pa.decimal128(3, scale=2))", "DECIMAL(10,5)", none_dataframe_value_str, none_expected_rows, False),

            # NaN

            ("dtype='uint8[pyarrow]'", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='uint16[pyarrow]'", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='uint32[pyarrow]'", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='uint64[pyarrow]'", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='int8[pyarrow]'", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='int16[pyarrow]'", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='int32[pyarrow]'", "integer", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='int64[pyarrow]'", "integer", nan_dataframe_value_str, nan_expected_rows, False),

            ("dtype='float16[pyarrow]'", "float", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='float32[pyarrow]'", "float", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='float64[pyarrow]'", "float", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='halffloat[pyarrow]'", "float", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='float[pyarrow]'", "float", nan_dataframe_value_str, nan_expected_rows, False),
            ("dtype='double[pyarrow]'", "float", nan_dataframe_value_str, nan_expected_rows, False),

            ("dtype='string[pyarrow]'", "VARCHAR(2000000)", nan_dataframe_value_str, nan_expected_rows, False),

            ("dtype='bool[pyarrow]'", "boolean", nan_dataframe_value_str, nan_expected_rows, False),

            #("dtype=pd.ArrowDtype(pa.timestamp('ns','UTC'))", "timestamp", nan_dataframe_value_str, nan_expected_rows, False), # Dateframe creation fails with: pyarrow.lib.ArrowNotImplementedError: Unsupported cast from double to timestamp using function cast_timestamp
            ("dtype=pd.ArrowDtype(pa.decimal128(3, scale=2))", "DECIMAL(10,5)", nan_dataframe_value_str, nan_expected_rows, False),
        ]

    @useData(types)
    def test_dtype_emit(self, dtype_definition:str, sql_type:str, dataframe_value_str:str, expected_result:List[Tuple], use_almost_equal:bool):
        sql=udf.fixindent(f'''
            CREATE OR REPLACE PYTHON3 SET SCRIPT test_dtype_emit(i integer) 
            EMITS (o1 {sql_type}, o2 {sql_type}, traceback varchar(2000000)) AS

            def run(ctx):
                try:
                    from decimal import Decimal
                    import pandas as pd
                    import numpy as np
                    import pyarrow as pa
                    from datetime import datetime, date
                    {dtype_definition}
                    df = pd.DataFrame({dataframe_value_str}, dtype=dtype)
                    df["traceback"]=None
                    ctx.emit(df)
                except:
                    import traceback
                    ctx.emit(None,None,traceback.format_exc())
            /
            ''')
        print(sql)
        self.query(sql)
        rows = self.query('''SELECT test_dtype_emit(0)''')
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

if __name__ == '__main__':
    udf.main()
