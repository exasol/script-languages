#!/usr/bin/env python2.7

import os
import sys
from decimal import Decimal
from datetime import date
from datetime import datetime

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class PandasDataFrame(udf.TestCase):
    def setUp(self):

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
        self.create_table_2()
        self.create_table_3()

    def create_table(self,table_name,create_col_defs_str):
        create_table_sql='CREATE TABLE %s (%s)' % (table_name,create_col_defs_str)
        print("Create Table Statement %s"%create_table_sql)
        self.query(create_table_sql)

    def create_table_1(self):
        self.create_table("TEST1",self.create_col_defs_str)
        self.import_via_insert("TEST1",[self.col_tuple],column_names=self.col_names)
        num_inserts = 9
        for i in range(num_inserts):
            insert_sql = 'INSERT INTO TEST1 (%s) SELECT %s FROM TEST1' % (self.col_names_str, self.col_names_str)
            print("Insert Statement %s"%insert_sql)
            self.query(insert_sql)
        self.num_rows = 2**num_inserts

    def create_table_2(self):
        self.create_table("TEST2",self.create_col_defs_str)
        self.col_tuple_1 = (
                Decimal('1'), 
                Decimal('1'), 
                Decimal('1'), 
                Decimal('1'), 
                Decimal('1'), 
                1, 
                True, 
                'abcdefghij', 
                'abcdefgh  ', 
                date(2018, 10, 12), 
                datetime(2018, 10, 12, 12, 15, 30, 123000)
                )
        self.import_via_insert("TEST2",[self.col_tuple_1],column_names=self.col_names)
        self.col_tuple_2 = (
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
        self.import_via_insert("TEST2",[self.col_tuple_2],column_names=self.col_names)
        self.col_tuple_null = (None, None, None, None, None, None, None, None, None, None, None) 
        self.import_via_insert("TEST2",[self.col_tuple_null],column_names=self.col_names)

    def create_table_3(self):
        self.create_col_defs_3 = [
            ('C0','INT IDENTITY'),
            ('C1','INTEGER'), 
            ]
        self.create_col_defs_str_3 = ','.join(
                '%s %s'%(name,type_decl) 
                for name, type_decl 
                in self.create_col_defs_3
                )
        self.col_defs_3 = self.create_col_defs_3[1:]
        self.col_defs_str_3 = ','.join(
                '%s %s'%(name,type_decl) 
                for name, type_decl 
                in self.col_defs_3
                )
        self.col_names_3 = [name for name, type_decl in self.col_defs_3]
        self.col_names_str_3 = ','.join(self.col_names_3)

        self.create_table("TEST3",self.create_col_defs_str_3)
        self.test3_num_rows = 10
        self.col_tuple_3 = [(i,) for i in range(self.test3_num_rows)]
        self.import_via_insert("TEST3",self.col_tuple_3,column_names=self.col_names_3)

    def test_dataframe_scalar_emits(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe()
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)

    def test_dataframe_scalar_returns(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            RETURNS DECIMAL(10,5) AS

            import numpy as np

            def run(ctx):
                df = ctx.get_dataframe()
                return np.asscalar(df.iloc[0, 0] + df.iloc[0, 1])
            /
            ''' % (self.col_defs_str))
        self.query(udf_sql)
        print(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([(Decimal('1235'),)]*self.num_rows, rows)

    def test_dataframe_scalar_emits_no_iter(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS
            
            def run(ctx):
                df = ctx.get_dataframe()
                df = ctx.get_dataframe()
                df = ctx.get_dataframe()
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)

    def test_dataframe_scalar_emits_col_names(self):
        output_columns = 'X1 VARCHAR(5), X2 VARCHAR(5), X3 VARCHAR(5), X4 VARCHAR(5), X5 VARCHAR(5), X6 VARCHAR(5), X7 VARCHAR(5), X8 VARCHAR(5), X9 VARCHAR(5), X10 VARCHAR(5), X11 VARCHAR(5)'
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe()
                ctx.emit(*(df.columns.tolist()))
            /
            ''' % (self.col_defs_str, output_columns))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([tuple(self.col_names)]*self.num_rows, rows)

    def test_dataframe_scalar_emits_unique(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(C0 INT)
            EMITS(C0 INT) AS
            import numpy as np

            def run(ctx):
                df = ctx.get_dataframe()
                ctx.emit(np.asscalar(df.C0))
            /
            ''')
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(C0) FROM FN2.TEST1'
        print(select_sql)
        rows = self.query(select_sql)
        self.assertEqual(self.num_rows, len(set([x[0] for x in rows])))

    def test_dataframe_scalar_emits_all_unique(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(C0 INT)
            EMITS(C0 INT) AS
            import numpy as np

            def run(ctx):
                df = ctx.get_dataframe(num_rows="all")
                ctx.emit(np.asscalar(df.C0))
            /
            ''')
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(C0) FROM FN2.TEST1'
        print(select_sql)
        rows = self.query(select_sql)
        self.assertEqual(self.num_rows, len(set([x[0] for x in rows])))

    def test_dataframe_scalar_emits_empty(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS
            import pandas as pd

            def run(ctx):
                df = pd.DataFrame()
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'emit DataFrame is empty'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_scalar_emits_wrong_args0(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS
            import pandas as pd

            def run(ctx):
                df = pd.DataFrame([[]])
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'emit\(\) takes exactly 11 arguments \(0 given\)'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_scalar_emits_wrong_args7(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe()
                df = df.iloc[:, 1:]
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'emit\(\) takes exactly 11 arguments \(10 given\)'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            
            def run(ctx):
                df = ctx.get_dataframe(num_rows="all")
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)

    def test_dataframe_set_returns(self):
        from decimal import Decimal
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            RETURNS DECIMAL(10,5) AS
            import numpy as np

            def run(ctx):
                df = ctx.get_dataframe(num_rows="all")
                return np.asscalar(df.iloc[:, 0].sum())
            /
            ''' % (self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([(Decimal(self.num_rows),)], rows)

    def test_dataframe_set_emits_iter(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            
            def run(ctx):
                while True:
                    df = ctx.get_dataframe(num_rows=1)
                    if df is None:
                        break
                    ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)

    def test_dataframe_set_emits_iter_getattr(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(R VARCHAR(1000)) AS
            def run(ctx):
                BATCH_ROWS = 1
                while True:
                    df = ctx.get_dataframe(num_rows=BATCH_ROWS)
                    if df is None:
                        break
                    ctx.emit(df.applymap(lambda x: "df_"+str(x)))
                    try:
                        ctx.emit("getattr_"+str(ctx.C1))
                        ctx.emit("eob") # end of batch
                    except:
                        ctx.emit("eoi") # end of iteration
            /
            ''' % (self.col_defs_str_3))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST3' % (self.col_names_str_3)
        print(select_sql)
        rows = self.query(select_sql)
        expected_result = [("df_"+str(self.col_tuple_3[0][0]),)]
        for i in range(1,self.test3_num_rows):
            expected_result.append(("getattr_"+str(self.col_tuple_3[i][0]),))
            expected_result.append(("eob",))
            expected_result.append(("df_"+str(self.col_tuple_3[i][0]),))
        expected_result.append(("eoi",))
        self.assertRowsEqual(expected_result, rows)

    def test_dataframe_set_emits_iter_exception(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            
            def run(ctx):
                while True:
                    df = ctx.get_dataframe(num_rows=1)
                    if df is None:
                        #break
                        df = ctx.get_dataframe(num_rows=1)
                    ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'Iteration finished'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_iter_reset_at_end(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            
            def run(ctx):
                i = 0
                while True:
                    df = ctx.get_dataframe(num_rows=3)
                    if df is None:
                        if i < 1:
                            ctx.reset()
                            i = i + 1
                        else:
                            break
                    else:
                        ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple]*self.num_rows*2, rows)

    def test_dataframe_set_emits_col_names(self):
        output_columns = 'X1 VARCHAR(5), X2 VARCHAR(5), X3 VARCHAR(5), X4 VARCHAR(5), X5 VARCHAR(5), X6 VARCHAR(5), X7 VARCHAR(5), X8 VARCHAR(5), X9 VARCHAR(5), X10 VARCHAR(5), X11 VARCHAR(5)'
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                while True:
                    df = ctx.get_dataframe(num_rows=1)
                    if df is None:
                        break
                    ctx.emit(*(df.columns.tolist()))
            /
            ''' % (self.col_defs_str, output_columns))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([tuple(self.col_names)]*self.num_rows, rows)

    def test_dataframe_set_emits_unique(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(C0 INT)
            EMITS(C0 INT) AS
            import numpy as np

            def run(ctx):
                while True:
                    df = ctx.get_dataframe(num_rows=1)
                    if df is None:
                        break
                    ctx.emit(np.asscalar(df.C0))
            /
            ''')
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(C0) FROM FN2.TEST1'
        print(select_sql)
        rows = self.query(select_sql)
        self.assertEqual(self.num_rows, len(set([x[0] for x in rows])))

    def test_dataframe_set_emits_all_unique(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(C0 INT)
            EMITS(C0 INT) AS
            import numpy as np

            def run(ctx):
                while True:
                    df = ctx.get_dataframe(num_rows="all")
                    if df is None:
                        break
                    for i in range(df.shape[0]):
                        ctx.emit(np.asscalar(df.iloc[i, 0]))
            /
            ''')
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(C0) FROM FN2.TEST1'
        print(select_sql)
        rows = self.query(select_sql)
        self.assertEqual(self.num_rows, len(set([x[0] for x in rows])))

    def test_dataframe_set_emits_empty(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            import pandas as pd

            def run(ctx):
                df = pd.DataFrame()
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'emit DataFrame is empty'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_wrong_args0(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            import pandas as pd

            def run(ctx):
                df = pd.DataFrame([[]])
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'emit\(\) takes exactly 11 arguments \(0 given\)'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_wrong_args7(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows="all")
                df = df.iloc[:, 1:]
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'emit\(\) takes exactly 11 arguments \(10 given\)'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_numrows_not_all(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows="some")
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'get_dataframe\(\) parameter'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_numrows_not_int(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows=True)
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'get_dataframe\(\) parameter'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_numrows_zero(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows=0)
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, 'get_dataframe\(\) parameter'):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_numrows_negative(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows=-1)
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, "get_dataframe\(\) parameter"):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST1' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_scalar_emits_null(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe()
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST2' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple_1, self.col_tuple_2, self.col_tuple_null], rows)

    def test_dataframe_set_emits_null(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows='all')
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST2' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple_1, self.col_tuple_2, self.col_tuple_null], rows)

    def test_dataframe_scalar_emits_start_col(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(start_col=2)
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, ','.join('%s %s'%t for t in self.col_defs[2:])))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST2' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple_1[2:], self.col_tuple_2[2:], self.col_tuple_null[2:]], rows)

    def test_dataframe_set_emits_null_start_col(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows='all', start_col=5)
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, ','.join('%s %s'%t for t in self.col_defs[5:])))
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(%s) FROM FN2.TEST2' % (self.col_names_str)
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual([self.col_tuple_1[5:], self.col_tuple_2[5:], self.col_tuple_null[5:]], rows)

    def test_dataframe_set_emits_null_start_col_negative(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows='all', start_col=-1)
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, "must be an integer >= 0"):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST2' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_null_start_col_too_large(self):
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows='all', start_col=100000)
                ctx.emit(df)
            /
            ''' % (self.col_defs_str, self.col_defs_str))
        print(udf_sql)
        self.query(udf_sql)
        with self.assertRaisesRegexp(Exception, "is 100000, but there are only %d input columns" % len(self.col_names)):
            select_sql = 'SELECT foo(%s) FROM FN2.TEST2' % (self.col_names_str)
            print(select_sql)
            rows = self.query(select_sql)

    def test_dataframe_set_emits_timestamp_only(self):
        import datetime
        udf_sql = udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SET SCRIPT foo(sec int) EMITS (ts timestamp) AS

            def run(ctx):
                import pandas as pd
                import numpy as np
                from datetime import datetime

                c1=np.empty(shape=(2),dtype=np.object_)

                c1[:]="2020-07-27 14:22:33.600699"

                df=pd.DataFrame({0:c1})
                df[0]=pd.to_datetime(df[0])

                ctx.emit(df)
            /
            ''')
        print(udf_sql)
        self.query(udf_sql)
        select_sql = 'SELECT foo(1)'
        print(select_sql)
        rows = self.query(select_sql)
        self.assertRowsEqual(
                [
                    (datetime.datetime(2020, 7, 27, 14, 22, 33, 600000),),
                    (datetime.datetime(2020, 7, 27, 14, 22, 33, 600000),)
                ], rows)



if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

