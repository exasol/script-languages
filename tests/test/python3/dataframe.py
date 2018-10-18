#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class PandasDataFrame(udf.TestCase):
    def setUp(self):
        from decimal import Decimal
        from datetime import date
        from datetime import datetime

        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2', ignore_errors=True)

        #self.col_names = 'C1, C2, C3, C4, C5, C6, C7, C8'
        #self.col_defs = 'C1 INT, C2 DECIMAL(36,5), C3 DOUBLE, C4 BOOLEAN, C5 DATE, C6 TIMESTAMP, C7 VARCHAR(500), C8 CHAR(10)'
        #self.col_vals = "1, 12345.6789, 12345.6789, TRUE, '2018-09-12', '2018-09-12 13:37:00.123', 'abcdefghij', 'abcdefgh'"
        #self.col_tuple = (Decimal('1'), Decimal('12345.6789'), 12345.6789, True, date(2018, 9, 12), datetime(2018, 9, 12, 13, 37, 0, 123000), 'abcdefghij', 'abcdefgh  ')

        self.col_names = 'C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11'
        self.col_defs = 'C1 Decimal(2,0), C2 Decimal(4,0), C3 Decimal(8,0), C4 Decimal(16,0), C5 Decimal(36, 0), C6 DOUBLE, C7 BOOLEAN, C8 VARCHAR(500), C9 CHAR(10), C10 DATE, C11 TIMESTAMP'
        self.col_vals = "12, 1234, 12345678, 1234567890123456, 123456789012345678901234567890123456, 12345.6789, TRUE, 'abcdefghij', 'abcdefgh', '2018-10-12', '2018-10-12 12:15:30.123'"
        self.col_tuple = (Decimal('12'), Decimal('1234'), Decimal('12345678'), Decimal('1234567890123456'), Decimal('123456789012345678901234567890123456'), 12345.6789, True, 'abcdefghij', 'abcdefgh  ', date(2018, 10, 12), datetime(2018, 10, 12, 12, 15, 30, 123000))

        self.query('CREATE TABLE TEST1(C0 INT IDENTITY, %s)' % (self.col_defs))
        self.query('INSERT INTO TEST1 (%s) VALUES (%s)' % (self.col_names, self.col_vals))
        num_inserts = 6
        for i in range(num_inserts):
            self.query('INSERT INTO TEST1 (%s) SELECT %s FROM TEST1' % (self.col_names, self.col_names))
        self.num_rows = 2**num_inserts

        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            DATAFRAME_EMITS_HELPER(...)
            EMITS(...) AS

            import numpy as np
            import pandas as pd

            def to_scalar(val):
                if isinstance(val, (np.number, np.bool_)):
                    return np.asscalar(val)
                elif isinstance(val, pd.tslib.Timestamp):
                    return val.to_pydatetime()
                else:
                    return val

            def np_numeric_as_scalar(x):
                return [to_scalar(x[i]) for i in range(0, len(x))]

            def transform_dataframe_to_list(x):
                ret = []
                for i in range(0, x.shape[1]):
                    if np.issubdtype(x.iloc[:, i].dtypes, np.number):
                        ret.append([np.asscalar(v) for v in x.iloc[:, i].values])
                    elif np.issubdtype(x.iloc[:, i].dtypes, np.bool_):
                        ret.append([np.asscalar(v) for v in x.iloc[:, i].values])
                    elif pd.core.dtypes.common.is_datetime_or_timedelta_dtype(x.iloc[:, i]):
                        ret.append([v.to_pydatetime() for v in x.iloc[:, i]])
                    else:
                        ret.append([v for v in x.iloc[:, i]])
                ret = [list(i) for i in zip(*ret)]
                return ret

            def run(ctx):
                df = ctx.get_dataframe()
                df_list = transform_dataframe_to_list(df)
                for i in range(0, df.shape[0]):
                    ctx.emit(*df_list[i])
                #for i in range(0, df.shape[0]):
                #    out_list = np_numeric_as_scalar(df.iloc[i, :])
                #    ctx.emit(*out_list)
            /
            '''))

    def test_dataframe_test_c_func_set(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            import sys
            sys.path.append('/exaudf')
            import decimal
            import datetime
            import pandas as pd
            import pyextdataframe

            def run(ctx):
                pyListList = pyextdataframe.get_dataframe(exa.meta, ctx, 100)
                df = pd.DataFrame(pyListList)
                #df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: decimal.Decimal(x))
                #df.iloc[:, 4] = df.iloc[:, 4].apply(lambda x: datetime.datetime.strptime(x, "%%Y-%%m-%%d"))
                #df.iloc[:, 4] = df.iloc[:, 4].apply(lambda x: datetime.date(x.year, x.month, x.day))
                #df.iloc[:, 5] = df.iloc[:, 5].apply(lambda x: datetime.datetime.strptime(x, "%%Y-%%m-%%d %%H:%%M:%%S.%%f"))
                #ctx.emit(df)

                numpyTypes = [str(x) for x in df.dtypes.values]
                pyextdataframe.emit_dataframe(exa.meta, ctx, df, numpyTypes)
            /
            ''' % (self.col_defs, self.col_defs)))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)


    def test_dataframe_scalar_emits(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            import sys
            sys.path.append('/exaudf')
            import pyextdataframe
            
            def run(ctx):
                #df = ctx.get_dataframe()
                pyListList = pyextdataframe.get_dataframe(exa.meta, ctx, 2)
                df = pd.DataFrame(pyListList)

                #ctx.emit(df)
                numpyTypes = [str(x) for x in df.dtypes.values]
                pyextdataframe.emit_dataframe(exa.meta, ctx, df, numpyTypes)
            /
            ''' % (self.col_defs, self.col_defs)))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)

"""
    def test_dataframe_scalar_returns(self):
        from decimal import Decimal
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(%s)
            RETURNS DECIMAL(10,5) AS

            def run(ctx):
                df = ctx.get_dataframe()
                return df.iloc[0, 0] + df.iloc[0, 1]
            /
            ''' % (self.col_defs)))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([(Decimal('12346.6789'),)]*self.num_rows, rows)

    def test_dataframe_scalar_emits_no_iter(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS
            
            def run(ctx):
                df = ctx.get_dataframe()
                df = ctx.get_dataframe()
                df = ctx.get_dataframe()
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)

    def test_dataframe_scalar_emits_col_names(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe()
                ctx.emit(*(df.columns.tolist()))
            /
            ''' % (self.col_defs, 'X1 VARCHAR(5), X2 VARCHAR(5), X3 VARCHAR(5), X4 VARCHAR(5), X5 VARCHAR(5), X6 VARCHAR(5), X7 VARCHAR(5), X8 VARCHAR(5)')))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([tuple(self.col_names.split(", "))]*self.num_rows, rows)

    def test_dataframe_scalar_emits_unique(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(C0 INT)
            EMITS(C0 INT) AS
            import numpy as np

            def run(ctx):
                df = ctx.get_dataframe()
                ctx.emit(np.asscalar(df.C0))
            /
            '''))
        rows = self.query('SELECT foo(C0) FROM FN2.TEST1')
        self.assertEqual(self.num_rows, len(set([x[0] for x in rows])))

    def test_dataframe_scalar_emits_all_unique(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(C0 INT)
            EMITS(C0 INT) AS
            import numpy as np

            def run(ctx):
                df = ctx.get_dataframe(num_rows="all")
                ctx.emit(np.asscalar(df.C0))
            /
            '''))
        rows = self.query('SELECT foo(C0) FROM FN2.TEST1')
        self.assertEqual(self.num_rows, len(set([x[0] for x in rows])))

    def test_dataframe_scalar_emits_empty(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS
            import pandas as pd

            def run(ctx):
                df = pd.DataFrame()
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'emit DataFrame is empty'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_scalar_emits_wrong_args0(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS
            import pandas as pd

            def run(ctx):
                df = pd.DataFrame([[]])
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'emit\(\) takes exactly 8 arguments \(0 given\)'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_scalar_emits_wrong_args7(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe()
                df = df.iloc[:, 1:]
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'emit\(\) takes exactly 8 arguments \(7 given\)'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_set_emits(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            
            def run(ctx):
                df = ctx.get_dataframe(num_rows="all")
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)

    def test_dataframe_set_returns(self):
        from decimal import Decimal
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            RETURNS DECIMAL(10,5) AS
            import numpy as np

            def run(ctx):
                df = ctx.get_dataframe(num_rows="all")
                return np.asscalar(df.iloc[:, 0].sum())
            /
            ''' % (self.col_defs)))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([(Decimal(self.num_rows),)], rows)

    def test_dataframe_set_emits_iter(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            
            def run(ctx):
                while True:
                    df = ctx.get_dataframe(num_rows=1)
                    if df is None:
                        break
                    ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)

    def test_dataframe_set_emits_iter_exception(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
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
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'Iteration finished'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_set_emits_col_names(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                while True:
                    df = ctx.get_dataframe(num_rows=1)
                    if df is None:
                        break
                    ctx.emit(*(df.columns.tolist()))
            /
            ''' % (self.col_defs, 'X1 VARCHAR(5), X2 VARCHAR(5), X3 VARCHAR(5), X4 VARCHAR(5), X5 VARCHAR(5), X6 VARCHAR(5), X7 VARCHAR(5), X8 VARCHAR(5)')))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([tuple(self.col_names.split(", "))]*self.num_rows, rows)

    def test_dataframe_set_emits_unique(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
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
            '''))
        rows = self.query('SELECT foo(C0) FROM FN2.TEST1')
        self.assertEqual(self.num_rows, len(set([x[0] for x in rows])))

    def test_dataframe_set_emits_all_unique(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
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
            '''))
        rows = self.query('SELECT foo(C0) FROM FN2.TEST1')
        self.assertEqual(self.num_rows, len(set([x[0] for x in rows])))

    def test_dataframe_set_emits_empty(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            import pandas as pd

            def run(ctx):
                df = pd.DataFrame()
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'emit DataFrame is empty'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_set_emits_wrong_args0(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS
            import pandas as pd

            def run(ctx):
                df = pd.DataFrame([[]])
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'emit\(\) takes exactly 8 arguments \(0 given\)'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_set_emits_wrong_args7(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows="all")
                df = df.iloc[:, 1:]
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'emit\(\) takes exactly 8 arguments \(7 given\)'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_set_emits_numrows_not_all(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows="some")
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'get_dataframe\(\) parameter'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_set_emits_numrows_not_int(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows=True)
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'get_dataframe\(\) parameter'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_set_emits_numrows_zero(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows=0)
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, 'get_dataframe\(\) parameter'):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))

    def test_dataframe_set_emits_numrows_negative(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(%s)
            EMITS(%s) AS

            def run(ctx):
                df = ctx.get_dataframe(num_rows=-1)
                ctx.emit(df)
            /
            ''' % (self.col_defs, self.col_defs)))
        with self.assertRaisesRegexp(Exception, "get_dataframe\(\) parameter"):
            rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
"""

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

