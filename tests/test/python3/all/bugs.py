#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.testcase import skip


class SUP7256(udf.TestCase):
    def setUp(self):
        self.query('create schema sup7256', ignore_errors=True)
        self.query(udf.fixindent('''
                CREATE OR REPLACE python3 scalar SCRIPT pandas_babies(x int)
                returns int as
                import pandas as pd
                def run(ctx):
                    # This is just some stuff from the pandas tutorial
                    # The inital set of baby names and bith rates
                    names = ['Bob','Jessica','Mary','John','Mel']
                    births = [968, 155, 77, 578, 973]
                    BabyDataSet = list(zip(names,births))
                    df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
                    return len(df)
                /
                '''))

    def test_basic_pandas(self):
        row = self.query("select pandas_babies(12)")[0]
        self.assertEqual(5,row[0])
        

class DWA19106(udf.TestCase):
    def setUp(self):
        self.query('CREATE SCHEMA DWA19106', ignore_errors=True)
        self.query(udf.fixindent('''
                CREATE OR REPLACE python3 SET SCRIPT blow_up(x varchar(15000))
                EMITS (x varchar(15000)) AS
                def run(ctx):
                    #ctx.emit(str(len(ctx.x)))
                    ctx.emit(str(len(ctx.x.decode('utf-8'))))
                /
                '''))

    @skip("String handling in Python 3 differs vastly from Python 2")
    def test_explode(self):
        row = self.query('select blow_up(c) from (select "$RANDOM"(varchar(15000),0,0,15000,15000) as c), (select * from (values 202,202,203,203,205,206,208,207,206,204,205,206,206,207,208,208,203,203,203,203,204,204,204,204,209,209,208,207,207,206,205,205,200,203,205,206,206,205,206,207,203,204,205,205,206,206,207,207,206,205,204,204,203,203,204,204,204,203,203,202,201,200,199,199,201,201,201,201,201,201,201,201,200,199,198,198,198,74,74,75,75,74,73,71,70,71,71,72,72,72,73,73,73,75,76,77,76,74,72,73,74,76,78,79,78,76,75,76,77,82,81,81,81,81,81,81,81,77,78,78,77,77,78,80,81,77,78,79,81,83,84,84,84,84,84,83,82,80,79,78,76,75,75,76,79,80,79,77,71,67,162,159,159,164,168,173,176,181,182,185,189,192,195,195,197,208,209,212,211,212,211,210,209,207,208,209,209,208,207,205,203,206,206,206,207,209,211,213,213,207,205,205,206,206,207,207,207,204,204,204,204,205,205,205,205,206,205,205,204,202,201,201,200,199,201,203,204,203,203,204,206,203,203,204,204,205,205,206,207,206,205,204,204,203,203,204,204,200,200,200,201,201,201,201,201,201,201,201,201,201,201,201,201,198,198,197,197,197,69,70,72,74,75,74,73,73,72,72,72,72,73,73,73,74,74,75,77,77,75,74,74,75) as p(x))')[0]
        self.assertEqual(u'15000',row[0])


class SPOT_XYZ(udf.TestCase):
    def setUp(self):
        self.query('CREATE SCHEMA SPOT_XYZ', ignore_errors=True)
        self.query(udf.fixindent('''
                create or replace python3 scalar script large_exception(n int)
                returns int as

                def run(ctx):
                    x = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n'
                    raise Exception(x*ctx.n)
                    return ctx.n
                /
                '''))

    def test_large_exception_msg(self):
        with self.assertRaisesRegex(Exception, 'VM error:'):
            self.query('SELECT SPOT_XYZ.large_exception(300000) FROM dual')


if __name__ == '__main__':
    udf.main()
