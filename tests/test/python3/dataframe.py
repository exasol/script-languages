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
        self.col_names = 'C1, C2, C3, C4, C5, C6, C7, C8'
        self.col_defs = 'C1 INT, C2 DECIMAL(36,5), C3 DOUBLE, C4 BOOLEAN, C5 DATE, C6 TIMESTAMP, C7 VARCHAR(500), C8 CHAR(10)'
        self.col_vals = "1, 12345.6789, 12345.6789, TRUE, '2018-09-12', '2018-09-12 13:37:00.123', 'abcdefghij', 'abcdefgh'"
        self.col_tuple = (Decimal('1'), Decimal('12345.6789'), 12345.6789, True, date(2018, 9, 12), datetime(2018, 9, 12, 13, 37, 0, 123000), 'abcdefghij', 'abcdefgh  ')
        self.query('CREATE TABLE TEST1(%s)' % (self.col_defs))
        self.query('INSERT INTO TEST1 VALUES (%s)' % (self.col_vals))
        num_inserts = 6
        for i in range(num_inserts):
            self.query('INSERT INTO TEST1 SELECT * FROM TEST1')
        self.num_rows = 2**num_inserts

    def test_dataframe_scalar(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(%s)
            EMITS(%s) AS

            from decimal import Decimal
            import numpy as np
            import pandas as pd

            def to_scalar(val):
                if isinstance(val, (np.number, np.bool_)):
                    return np.asscalar(val)
                elif isinstance(val, pd.tslib.Timestamp):
                    #return val.astype('str')
                    return val.to_pydatetime()
                else:
                    return val

            def np_numeric_as_scalar(x):
                return [to_scalar(x[i]) for i in range(0, len(x))]
                #return [np.asscalar(x[i]) if isinstance(x[i], (np.number, np.bool_)) else x[i] for i in range(0, len(x))]

            def run(ctx):
                df = ctx.get_dataframe()
                for i in range(0, df.shape[0]):
                    out_list = np_numeric_as_scalar(df.iloc[i, :])
                    #out_list = df.iloc[i, :]
                    ctx.emit(*out_list)
            /
            ''' % (self.col_defs, self.col_defs)))
        rows = self.query('SELECT foo(%s) FROM FN2.TEST1' % (self.col_names))
        self.assertRowsEqual([self.col_tuple]*self.num_rows, rows)



"""
    def test_dataframe_set(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(C1 INT, C2 INT)
            EMITS(C1 INT, C2 INT) AS

            import numpy as np

            def np_numeric_as_scalar(x):
                return [np.asscalar(x[i]) for i in range(0, len(x))]

            def run(ctx):
                df = ctx.get_dataframe(num_rows=2)
                for i in range(0, df.shape[0]):
                    out_list = np_numeric_as_scalar(df.iloc[i, :])
                    ctx.emit(*out_list)
            /
            '''))
        rows = self.query('SELECT foo(C1, C2) FROM FN2.TEST1')
        self.assertRowsEqual([(1, 2), (3, 4)], rows)

    def test_dataframe_scalar(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SCALAR SCRIPT
            foo(C1 INT, C2 INT)
            EMITS(C1 INT, C2 INT) AS

            import numpy as np

            def np_numeric_as_scalar(x):
                return [np.asscalar(x[i]) for i in range(0, len(x))]

            def run(ctx):
                df = ctx.get_dataframe(num_rows=2)
                df = ctx.get_dataframe(num_rows=2)
                for i in range(0, df.shape[0]):
                    out_list = np_numeric_as_scalar(df.iloc[i, :])
                    ctx.emit(*out_list)
            /
            '''))
        rows = self.query('SELECT foo(C1, C2) FROM FN2.TEST1')
        self.assertRowsEqual([(1, 2), (3, 4)], rows)


    def test_get_dataframe_set_get_dataframe_null(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(C1 INT, C2 INT)
            EMITS(C1 INT, C2 INT) AS
            def run(ctx):
                df = ctx.get_dataframe(num_rows=2)
                df = ctx.get_dataframe(num_rows=2)
                if df is None:
                    ctx.emit(3, 42)
            /
            '''))
        rows = self.query('SELECT foo(C1, C2) FROM FN2.TEST1')
        self.assertRowsEqual([(3, 42)], rows)

    def test_get_dataframe_set_get_dataframe_exception(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT
            foo(C1 INT, C2 INT)
            EMITS(C1 INT, C2 INT) AS

            import numpy as np

            def np_numeric_as_scalar(x):
                return [np.asscalar(x[i]) for i in range(0, len(x))]

            def run(ctx):
                df = ctx.get_dataframe(num_rows=2)
                df = ctx.get_dataframe()
                df = ctx.get_dataframe()
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'Iteration finished'):
            rows = self.query('SELECT foo(C1, C2) FROM FN2.TEST1')
"""

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

