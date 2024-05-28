#!/usr/bin/env python3

from decimal import Decimal
from datetime import date
from datetime import datetime

from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from typing import List, Tuple, Union


class PandasDataFrameEmitDTypesMemoryLeakCheck(udf.TestCase):
    """
    This test creates huge dataframes inside a UDF, emits it,
    and validates that memory consumption is within expected range.
    """

    def setUp(self):
        self.maxDiff = None

        self.query(f'CREATE SCHEMA {self.__class__.__name__}', ignore_errors=True)
        self.query(f'OPEN SCHEMA {self.__class__.__name__}', ignore_errors=True)

    int_dataframe_value_str = "1"

    float16_dataframe_value_str = '1.1'
    float_dataframe_value_str = "1.1"

    str_dataframe_value_str = "'a'"

    bool_dataframe_value_str = "True"

    decimal_dataframe_value_str = "Decimal('1.1')"

    timestamp_dataframe_value_str = 'pd.Timestamp(datetime(2020, 7, 27, 14, 22, 33, 673251))'
    datetime_dataframe_value_str = 'datetime(2020, 7, 27, 14, 22, 33, 673251)'
    date_dataframe_value_str = 'date(2020, 7, 27)'

    types = [
        # Full columns without None or NaN / Int

        ("uint8", "integer", int_dataframe_value_str, 200000),
        ("uint16", "integer", int_dataframe_value_str, 200000),
        ("uint32", "integer", int_dataframe_value_str, 200000),
        ("uint64", "integer", int_dataframe_value_str, 200000),
        ("int8", "integer", int_dataframe_value_str, 200000),
        ("int16", "integer", int_dataframe_value_str, 200000),
        ("int32", "integer", int_dataframe_value_str, 200000),
        ("int64", "integer", int_dataframe_value_str, 200000),
        ("object", "integer", int_dataframe_value_str, 200000),

        # Full columns without None or NaN / Float

        ("float16", "double", float16_dataframe_value_str, 200000),
        ("float32", "double", float_dataframe_value_str, 200000),
        ("float64", "double", float_dataframe_value_str, 200000),
        ("float", "double", float_dataframe_value_str, 200000),
        ("double", "double", float_dataframe_value_str, 200000),
        ("object", "double", float_dataframe_value_str, 200000),

        # Full columns without None or NaN / Int to Float

        ("uint8", "double", int_dataframe_value_str, 200000),
        ("uint16", "double", int_dataframe_value_str, 200000),
        ("uint32", "double", int_dataframe_value_str, 200000),
        ("uint64", "double", int_dataframe_value_str, 200000),
        ("int8", "double", int_dataframe_value_str, 200000),
        ("int16", "double", int_dataframe_value_str, 200000),
        ("int32", "double", int_dataframe_value_str, 200000),
        ("int64", "double", int_dataframe_value_str, 200000),
        ("object", "double", int_dataframe_value_str, 200000),

        # Full columns without None or NaN / Float to Int

        ("float16", "integer", float16_dataframe_value_str, 200000),
        ("float32", "integer", float_dataframe_value_str, 200000),
        ("float64", "integer", float_dataframe_value_str, 200000),
        ("float", "integer", float_dataframe_value_str, 200000),
        ("double", "integer", float_dataframe_value_str, 200000),
        ("object", "integer", float_dataframe_value_str, 200000),

        # Full columns without None or NaN / Int to Decimal

        ("uint8", "DECIMAL(10,5)", int_dataframe_value_str, 200000),
        ("uint16", "DECIMAL(10,5)", int_dataframe_value_str, 200000),
        ("uint32", "DECIMAL(10,5)", int_dataframe_value_str, 200000),
        ("uint64", "DECIMAL(10,5)", int_dataframe_value_str, 200000),
        ("int8", "DECIMAL(10,5)", int_dataframe_value_str, 200000),
        ("int16", "DECIMAL(10,5)", int_dataframe_value_str, 200000),
        ("int32", "DECIMAL(10,5)", int_dataframe_value_str, 200000),
        ("int64", "DECIMAL(10,5)", int_dataframe_value_str, 200000),
        ("object", "DECIMAL(10,5)", int_dataframe_value_str, 200000),

        # Full columns without None or NaN / Float to Decimal

        ("float16", "DECIMAL(10,5)", float16_dataframe_value_str, 200000),
        ("float32", "DECIMAL(10,5)", float_dataframe_value_str, 200000),
        ("float64", "DECIMAL(10,5)", float_dataframe_value_str, 200000),
        ("float", "DECIMAL(10,5)", float_dataframe_value_str, 200000),
        ("double", "DECIMAL(10,5)", float_dataframe_value_str, 200000),
        ("object", "DECIMAL(10,5)", float_dataframe_value_str, 200000),

        # Full columns without None or NaN / Decimal

        ("object", "DECIMAL(10,5)", decimal_dataframe_value_str, 200000),

        # Full columns without None or NaN / String

        ("string", "VARCHAR(2000000)", str_dataframe_value_str, 200000),
        ("object", "VARCHAR(2000000)", str_dataframe_value_str, 200000),

        # Full columns without None or NaN / Boolean

        ("bool_", "boolean", bool_dataframe_value_str, 200000),
        ("boolean", "boolean", bool_dataframe_value_str, 200000),
        ("object", "boolean", bool_dataframe_value_str, 200000),

        # Full columns without None or NaN / Date and Time

        ("datetime64[ns]", "timestamp", timestamp_dataframe_value_str, 200000),
        ("object", "timestamp", timestamp_dataframe_value_str, 200000),
        ("object", "DATE", date_dataframe_value_str, 200000),
    ]

    @useData(types)
    def test_dtype_emit(self, dtype: str, sql_type: str, dataframe_value_str: str, max_memory: int):
        emit_cols = [f"o{i} {sql_type}" for i in range(25)]
        emit_cols_str = ",".join(emit_cols)
        udf_def_str = udf.fixindent(f'''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT test_dtype_emit("batch_size" integer, "batch_count" integer) 
            EMITS ({emit_cols_str}) AS

            import gc
            import tracemalloc
            import pandas as pd
            from decimal import Decimal
            from datetime import datetime, date

            def run(ctx):
                tracemalloc.start()
                for i in range(ctx.batch_count):
                    df = pd.DataFrame([[{dataframe_value_str} for c in range(25)] for r in range(ctx.batch_size)], 
                                      dtype="{dtype}")
                    ctx.emit(df)
                    if i == 0:
                        snapshot_begin = tracemalloc.take_snapshot()
                    elif i == ctx.batch_count - 1:
                        snapshot_end = tracemalloc.take_snapshot()
                        top_stats_begin_end = snapshot_end.compare_to(snapshot_begin, 'lineno')
                        first_item = top_stats_begin_end[0] #First item is always the largest one
                        if first_item.size_diff > {max_memory}:
                            raise RuntimeError(f"scalar emit UDF uses too much memory: {{first_item}}")             

            /
            ''')
        print(udf_def_str)
        self.query(udf_def_str)
        rows = self.query('''SELECT test_dtype_emit(100, 1000)''')
        assert len(rows[0]) == 25
        assert len(rows) == 100000


if __name__ == '__main__':
    udf.main()
