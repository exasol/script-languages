#!/usr/bin/env python3
import textwrap
from decimal import Decimal
from datetime import date
from datetime import datetime

from  exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import useData
from exasol_python_test_framework.udf.udf_debug import UdfDebugger
from typing import List, Tuple, Union

NUM_ROW_INSERTS = 17
NUM_ROWS = 2**NUM_ROW_INSERTS # => ~128 K rows

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

        for i in range(NUM_ROW_INSERTS):
            insert_sql = 'INSERT INTO TEST1 (%s) SELECT %s FROM TEST1' % (self.col_names_str, self.col_names_str)
            print("Insert Statement %s"%insert_sql)
            self.query(insert_sql)


    def test_dataframe_scalar_emits(self):
        """
        This test checks that the largest memory block of a tracemalloc snapshot diff is not larger than NUM_ROWS=131072 Bytes,
        where the memory block snapshots are retrieved during the first/last invocation of the scalar UDF,
        but after the emit().
        Reasoning for NUM_ROWS=131072 Bytes is, if there was 1 Byte leaking during every execution,
        it would be found here.
        """
        udf_def_str = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            %%perNodeAndCallInstanceLimit 1;

            import tracemalloc
            import gc
            import time
            snapshot_begin = None
            memory_check_executed = False
            tracemalloc.start()
            counter = 0

            def run(ctx):
                df = ctx.get_dataframe()
                global memory_check_executed
                global snapshot_begin
                global counter
                ctx.emit(df)
                if counter == 0:
                    print("Retrieving start snapshot", flush=True)
                    snapshot_begin = tracemalloc.take_snapshot()
                if counter == %s:
                    assert memory_check_executed == False #Sanity check for row number
                    print("Checking memory usage", flush=True)
                    gc.collect()
                    time.sleep(1) #Need to sleep here. Otherwise GC might not have finished before taking snapshot.
                    snapshot_end = tracemalloc.take_snapshot()
                    top_stats_begin_end = snapshot_end.compare_to(snapshot_begin, 'lineno')
                    first_item = top_stats_begin_end[0] #First item is always the largest one
                    if first_item.size_diff > %s:
                        raise RuntimeError(f"scalar emit UDF uses too much memory: {first_item}")
                    memory_check_executed = True
                counter = counter + 1
            /

        ''' % (self.col_defs_str, self.col_defs_str, NUM_ROWS - 1, NUM_ROWS))
        self.query(udf_def_str)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        rows = self.query(select_sql)
        self.assertEqual(NUM_ROWS, len(rows))

    def test_dataframe_scalar_emits_multiple(self):
        """
        This test checks that the largest memory block of a tracemalloc snapshot diff is not larger than 150KB, where
        the memory block snapshots are retrieved during the first/last invocation of the scalar UDF,
        but after the emit(). The emit is called multiple times in this test. The scalar UDF gets 2^3 = 8 rows as input,
        and emits 2~14 = 16384 dataframes of row size = 1 in each execution.
        For some reason
        (which would require further investigation) this test allocates ~ 120KB of memory in the <EXASCRIPT> module.
        (Which is more compared to the other tests in this file).
        However, manual tests showed that a further increase of the number of emitted dataframes (2~17 * 5)
        does not increase the allocated memory even more.
        """
        batch_count = 8
        batch_size = int(NUM_ROWS / batch_count)
        udf_def_str = udf.fixindent(f'''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo({self.col_defs_str})
            EMITS({self.col_defs_str}) AS

            %perNodeAndCallInstanceLimit 1;

            import tracemalloc
            import gc
            import time
            snapshot_begin = None
            memory_check_executed = False
            tracemalloc.start()
            counter = 0

            def run(ctx):
                df = ctx.get_dataframe()
                assert df.shape[0] == 1
                global memory_check_executed
                global snapshot_begin
                global counter
                for idx_batch in range({batch_size}):
                    ctx.emit(df)
                    if counter == 0 and idx_batch == 0:
                        print("Retrieving start snapshot", flush=True)
                        snapshot_begin = tracemalloc.take_snapshot()
                if counter == {batch_count - 1}:
                    assert memory_check_executed == False #Sanity check for row number
                    print("Checking memory usage", flush=True)
                    gc.collect()
                    time.sleep(1) #Need to sleep here. Otherwise GC might not have finished before taking snapshot.
                    snapshot_end = tracemalloc.take_snapshot()
                    top_stats_begin_end = snapshot_end.compare_to(snapshot_begin, 'lineno')
                    first_item = top_stats_begin_end[0] #First item is always the largest one
                    if first_item.size_diff > 170000:
                        raise RuntimeError(f"scalar emit UDF uses too much memory: {{first_item}}")
                    memory_check_executed = True
                counter = counter + 1
            /

        ''')
        print(udf_def_str)
        self.query(udf_def_str)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1 WHERE C0 <= %d; ' % (self.col_names_str, batch_count)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertEqual(NUM_ROWS, len(rows))

    def test_dataframe_scalar_returns(self):
        """
        This test checks that the largest memory block of a tracemalloc snapshot diff is not larger than NUM_ROWS = 131072 Bytes,
        where the memory block snapshots are retrieved during the first/last invocation of the scalar UDF.
        Reasoning for NUM_ROWS=131072 Bytes is, if there was 1 Byte leaking during every execution,
        it would be found here.
        """
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            RETURNS DECIMAL(10,5) AS


            %%perNodeAndCallInstanceLimit 1;

            import tracemalloc
            import gc
            import time
            snapshot_begin = None
            memory_check_executed = False
            tracemalloc.start()
            counter = 0

            def run(ctx):
                df = ctx.get_dataframe()
                global memory_check_executed
                global snapshot_begin
                global counter
                if counter == 0:
                    print("Retrieving start snapshot", flush=True)
                    snapshot_begin = tracemalloc.take_snapshot()
                if counter == %s:
                    assert memory_check_executed == False #Sanity check for row number
                    print("Checking memory usage", flush=True)
                    gc.collect()
                    time.sleep(1) #Need to sleep here. Otherwise GC might not have finished before taking snapshot.
                    snapshot_end = tracemalloc.take_snapshot()
                    top_stats_begin_end = snapshot_end.compare_to(snapshot_begin, 'lineno')
                    first_item = top_stats_begin_end[0] #First item is always the largest one
                    if first_item.size_diff > %s:
                        raise RuntimeError(f"scalar returns UDF uses too much memory: {first_item}")
                    memory_check_executed = True
                counter = counter + 1

                return (df.iloc[0, 0] + df.iloc[0, 1]).item()
            /
            ''' % (self.col_defs_str, NUM_ROWS - 1, NUM_ROWS))
        self.query(udf_sql)
        print(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertEqual(NUM_ROWS, len(rows))


    def test_dataframe_set_emits(self):
        """
        This test checks that the largest memory block of a tracemalloc snapshot diff is not larger than 20000Bytes,
        where the memory block snapshots are retrieved during the first/last get+emit
        of a batch of 2^14 = 16384 rows. In total, 2^3 = 8 batches are retrieved and emitted.
        """
        batch_count = 8
        batch_size = int(NUM_ROWS / batch_count)
        udf_sql = udf.fixindent(f'''
                    CREATE OR REPLACE PYTHON3 SET SCRIPT
                    foo({self.col_defs_str})
                    EMITS({self.col_defs_str}) AS

                    import tracemalloc
                    import gc
                    import time
                    tracemalloc.start()

                    def process_df(ctx):
                        df = ctx.get_dataframe(num_rows={batch_size})
                        ctx.emit(df)

                    def run(ctx):

                        for batch_idx in range({batch_count}):
                            print(f"Processing batch {{batch_idx}}", flush=True)

                            process_df(ctx)
                            if batch_idx == 0:
                                gc.collect()
                                snapshot_begin = tracemalloc.take_snapshot()

                            if batch_idx == ({batch_count} - 1):
                                gc.collect()
                                time.sleep(1) #Need to sleep here. Otherwise GC might not have finished before taking snapshot.
                                snapshot_end = tracemalloc.take_snapshot()
                                top_stats_begin_end = snapshot_end.compare_to(snapshot_begin, 'lineno')
                                first_item = top_stats_begin_end[0] #First item is always the largest one
                                print(f"Largest memory item is {{first_item}}", flush=True)
                                if first_item.size_diff > 20000:
                                    raise RuntimeError(f"scalar emit UDF uses too much memory: {{first_item}}")
                    /
                    ''')
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM TEST1' % self.col_names_str
        print(select_sql)
        rows = self.query(select_sql)
        self.assertEqual(NUM_ROWS, len(rows))


if __name__ == '__main__':
    udf.main()

