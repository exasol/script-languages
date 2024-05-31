#!/usr/bin/env python3
import textwrap
from decimal import Decimal
from datetime import date
from datetime import datetime

from  exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from typing import List, Tuple, Union


class PandasDataFrameMemoryLeakTest(udf.TestCase):
    def setUp(self):
        self.maxDiff=None

        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2', ignore_errors=True)

        self.create_col_defs = [
            ('C0','INT IDENTITY'),
            ('C1','Decimal(2,0)'), 
            ('C2','Decimal(4,0)'),
            ('C3','Decimal(8,0)'),
            ('C4','Decimal(16,0)'),
            ('C5','Decimal(36,0)'),
            ('C6','DOUBLE'),
            ('C7','BOOLEAN'),
            ('C8','VARCHAR(500)'),
            ('C9','CHAR(10)'),
            ('C10','DATE'),
            ('C11','TIMESTAMP')
            ]
        self.create_col_defs_str = ','.join(
                '%s %s'%(name,type_decl) 
                for name, type_decl 
                in self.create_col_defs
                )
        self.col_defs = self.create_col_defs[1:]
        self.col_defs_str = ','.join(
                '%s %s'%(name,type_decl) 
                for name, type_decl 
                in self.col_defs
                )
        self.col_names = [name for name, type_decl in self.col_defs]
        self.col_names_str = ','.join(self.col_names)

        self.col_tuple = (
                Decimal('1'), 
                Decimal('1234'), 
                Decimal('12345678'), 
                Decimal('1234567890123456'), 
                Decimal('123456789012345678901234567890123456'), 
                12345.6789, 
                True, 
                'abcdefghij', 
                'abcdefgh  ', 
                date(2018, 10, 12), 
                datetime(2018, 10, 12, 12, 15, 30, 123000)
                )
        self.create_table_1()


    def create_table(self,table_name,create_col_defs_str):
        create_table_sql='CREATE TABLE %s (%s)' % (table_name,create_col_defs_str)
        print("Create Table Statement %s"%create_table_sql)
        self.query(create_table_sql)

    def create_table_1(self):
        self.create_table("TEST1",self.create_col_defs_str)
        self.import_via_insert("TEST1",[self.col_tuple],column_names=self.col_names)
        num_inserts = 17 # => ~128 K rows
        for i in range(num_inserts):
            insert_sql = 'INSERT INTO TEST1 (%s) SELECT %s FROM TEST1' % (self.col_names_str, self.col_names_str)
            print("Insert Statement %s"%insert_sql)
            self.query(insert_sql)
        self.num_rows = 2**num_inserts

    # def test_dataframe_scalar_emits(self):
    #     """
    #     This test checks that the largest memory block of a tracemalloc snapshot diff is not larger than 100KB, where
    #     the memory block snapshots are retrieved during the first/last invocation of the scalar UDF,
    #     but after the emit().
    #     Reasoning for 100KB is that the number of rows is > 100K, so if there was 1 Byte leaking during every execution,
    #     it would be found here.
    #     """
    #     udf_def_str = udf.fixindent('''
    #         CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
    #         foo(%s)
    #         EMITS(%s) AS
    #
    #         %%perNodeAndCallInstanceLimit 1;
    #
    #         import tracemalloc
    #         import gc
    #         snapshot_begin = None
    #         memory_check_executed = False
    #         tracemalloc.start()
    #         counter = 0
    #
    #         def run(ctx):
    #             df = ctx.get_dataframe()
    #             global memory_check_executed
    #             global snapshot_begin
    #             global counter
    #             ctx.emit(df)
    #             if counter == 0:
    #                 print("Retrieving start snapshot", flush=True)
    #                 snapshot_begin = tracemalloc.take_snapshot()
    #             if counter == %s:
    #                 assert memory_check_executed == False #Sanity check for row number
    #                 print("Checking memory usage", flush=True)
    #                 gc.collect()
    #                 snapshot_end = tracemalloc.take_snapshot()
    #                 top_stats_begin_end = snapshot_end.compare_to(snapshot_begin, 'lineno')
    #                 first_item = top_stats_begin_end[0] #First item is always the largest one
    #                 if first_item.size_diff > 100000:
    #                     raise RuntimeError(f"scalar emit UDF uses too much memory: {first_item}")
    #                 memory_check_executed = True
    #             counter = counter + 1
    #         /
    #
    #     ''' % (self.col_defs_str, self.col_defs_str, self.num_rows - 1))
    #     self.query(udf_def_str)
    #     select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
    #     rows = self.query(select_sql)
    #     self.assertEqual(self.num_rows, len(rows))
    #
    # def test_dataframe_scalar_returns(self):
    #     """
    #     This test checks that the largest memory block of a tracemalloc snapshot diff is not larger than 100KB, where
    #     the memory block snapshots are retrieved during the first/last invocation of the scalar UDF.
    #     Reasoning for 100KB is that the number of rows is > 100K, so if there was 1 Byte leaking during every execution,
    #     it would be found here.
    #     """
    #     udf_sql = udf.fixindent('''
    #         CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
    #         foo(%s)
    #         RETURNS DECIMAL(10,5) AS
    #
    #
    #         %%perNodeAndCallInstanceLimit 1;
    #
    #         import tracemalloc
    #         import gc
    #         snapshot_begin = None
    #         memory_check_executed = False
    #         tracemalloc.start()
    #         counter = 0
    #
    #         def run(ctx):
    #             df = ctx.get_dataframe()
    #             global memory_check_executed
    #             global snapshot_begin
    #             global counter
    #             if counter == 0:
    #                 print("Retrieving start snapshot", flush=True)
    #                 snapshot_begin = tracemalloc.take_snapshot()
    #             if counter == %s:
    #                 assert memory_check_executed == False #Sanity check for row number
    #                 print("Checking memory usage", flush=True)
    #                 gc.collect()
    #                 snapshot_end = tracemalloc.take_snapshot()
    #                 top_stats_begin_end = snapshot_end.compare_to(snapshot_begin, 'lineno')
    #                 first_item = top_stats_begin_end[0] #First item is always the largest one
    #                 if first_item.size_diff > 100000:
    #                     raise RuntimeError(f"scalar emit UDF uses too much memory: {first_item}")
    #                 memory_check_executed = True
    #             counter = counter + 1
    #
    #             return (df.iloc[0, 0] + df.iloc[0, 1]).item()
    #         /
    #         ''' % (self.col_defs_str, self.num_rows - 1))
    #     self.query(udf_sql)
    #     print(udf_sql)
    #     select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
    #     print(select_sql)
    #     rows = self.query(select_sql)
    #     self.assertEqual(self.num_rows, len(rows))


    def test_dataframe_set_emits(self):
        """
        This test validates that the <EXASCRIPT> module does not leak more than 100kb of RAM in a set/emits UDF.
        The test is different from the others as it checks only the <EXASCRIPT> for leaks,
        not the rest of the UDF client; the reason for that is that a
        set UDF reads all input rows at once into a dataframe.
        <EXASCRIPT> is the module name
        of the pyextdataframe.so library (named during the runtime compilation during execution of a Python UDF).
        """
        batch_count = 10
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            import tracemalloc
            import gc
            tracemalloc.start()
            counter = 0
            end_reached = False

            def process_df(ctx):            
                df = ctx.get_dataframe(num_rows=%d)
                ctx.emit(df)

            def run(ctx):
                global counter
                global end_reached
                assert end_reached == False
                process_df(ctx)
                if counter == 0:
                    gc.collect()
                    snapshot_begin = tracemalloc.take_snapshot()

                if counter == (%d - 1):
                    end_reached = True
                    gc.collect()
                    snapshot_end = tracemalloc.take_snapshot()
                    top_stats_begin_end = snapshot_end.compare_to(snapshot_begin, 'lineno')
                    first_item = top_stats_begin_end[0] #First item is always the largest one
                    if first_item.size_diff > 100000:
                        raise RuntimeError(f"scalar emit UDF uses too much memory: {first_item}")

            /
            ''' % (self.col_defs_str, self.col_defs_str, self.num_rows, self.num_rows))
        print(udf_sql)
        self.query(udf_sql)
        select_str = "SELECT * FROM FN2.TEST1\n"

        union_str = "UNION ALL\n".join([select_str for i in range(batch_count)])

        select_sql = '''
WITH union_cte AS (
%s
)
SELECT foo(%s) FROM union_cte
        ''' % (union_str, self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertEqual(self.num_rows * batch_count, len(rows))


if __name__ == '__main__':
    udf.main()

