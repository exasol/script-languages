#!/usr/bin/env python3

import datetime

from exasol_python_test_framework import udf


class _JavaUdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            create java scalar script dob_1i_1o(x double) emits(y double)
            as
            class DOB_1I_1O{
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x"));
                    ctx.emit(ctx.getDouble("x"));
                    }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script line_1i_1o(x double) emits(y double)
            as
            class LINE_1I_1O{
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x"));
                    }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script line_1i_2o(x double) emits(y double, z double)
            as
            class LINE_1I_2O{
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x"),ctx.getDouble("x"));
                    }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script line_2i_1o(x double, y double) emits(z double)
            as
            class LINE_2I_1O{
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x")+ctx.getDouble("y"));
                    }
            }
            /
        '''))

        self.query(udf.fixindent('''
            create java scalar script line_3i_2o(x double, y double, z double) emits(z1 double, z2 double)
            as
            class LINE_3I_2O{
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x")+ctx.getDouble("y"),3000);
                    }
            }
            /
        '''))

class InputOutputMatchingTest(_JavaUdfSetup):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT dob_1i_1o(x double)
            EMITS (y double) AS
            class DOB_1I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"));
                    ctx.emit(ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_1i_1o(x double)
            EMITS (y double) AS
            class LINE_1I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_1i_2o(x double)
            EMITS (y double, z double) AS
            class LINE_1I_2O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"), ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_2i_1o(x double, y double)
            EMITS (z double) AS
            class LINE_2I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x") + ctx.getDouble("y"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_3i_2o(x double, y double, z double)
            EMITS (z1 double, z2 double) AS
            class LINE_3I_2O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x") + ctx.getDouble("y"), 3000.0);
                }
            }
            /
        '''))
        
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create table fn2.t(id double, x double)')
        self.query('insert into fn2.t values (100,1),(100,2),(200,3)')

    def test_iomatch_1i_1o(self):
        rows = self.query('''
            select x*2, fn1.line_1i_1o(x), x*3
            FROM fn2.t
            ''')
        self.assertRowsEqual(sorted([(2,1,3,), (6,3,9,), (4,2,6,)]), sorted(rows))

    def test_iomatch_1i_2o(self):
        rows = self.query('''
            select x*2, fn1.line_1i_2o(x), x*3
            FROM fn2.t
            ''')
        self.assertRowsEqual(sorted([(2,1,1,3,), (6,3,3,9,), (4,2,2,6,)]), sorted(rows))

    def test_iomatch_2i_1o(self):
        rows = self.query('''
            select x*2, fn1.line_2i_1o(x,id), x*3
            FROM fn2.t
            ''')
        self.assertRowsEqual(sorted([(2,101,3,), (6,203,9,), (4,102,6,)]), sorted(rows))

    def test_iomatch_3i_2o(self):
        rows = self.query('''
            select x*2, fn1.line_3i_2o(x,id,id), x*3
            FROM fn2.t
            ''')
        self.assertRowsEqual(sorted([(2,101,3000,3,), (6,203,3000,9,), (4,102,3000,6,)]), sorted(rows))

    def test_iomatch_dob_1i_1o(self):
        rows = self.query('''
            select x*2, fn1.dob_1i_1o(x), x*3
            FROM fn2.t
            ''')
        self.assertRowsEqual(sorted([(2,1,3,), (2,1,3,), (6,3,9,), (6,3,9,), (4,2,6,),  (4,2,6,)]), sorted(rows))


class ColumnNamesTest(_JavaUdfSetup):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT dob_1i_1o(x double)
            EMITS (y double) AS
            class DOB_1I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"));
                    ctx.emit(ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_1i_1o(x double)
            EMITS (y double) AS
            class LINE_1I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_1i_2o(x double)
            EMITS (y double, z double) AS
            class LINE_1I_2O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"), ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_2i_1o(x double, y double)
            EMITS (z double) AS
            class LINE_2I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x") + ctx.getDouble("y"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_3i_2o(x double, y double, z double)
            EMITS (z1 double, z2 double) AS
            class LINE_3I_2O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x") + ctx.getDouble("y"), 3000.0);
                }
            }
            /
        '''))
        
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create table fn2.t(id double, x double)')
        self.query('insert into fn2.t values (100,1),(100,2),(200,3)')

    def test_col_names(self):
        self.query('''
            create or replace table fn2.foo as select x*2 a, fn1.line_3i_2o(x,id,id), x*3 b
            FROM fn2.t
            ''')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('A',rows[0][0])
        self.assertEqual('Z1',rows[1][0])
        self.assertEqual('Z2',rows[2][0])
        self.assertEqual('B',rows[3][0])


class DatatypesTest(_JavaUdfSetup):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT dob_1i_1o(x double)
            EMITS (y double) AS
            class DOB_1I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"));
                    ctx.emit(ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_1i_1o(x double)
            EMITS (y double) AS
            class LINE_1I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_1i_2o(x double)
            EMITS (y double, z double) AS
            class LINE_1I_2O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"), ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_2i_1o(x double, y double)
            EMITS (z double) AS
            class LINE_2I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x") + ctx.getDouble("y"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_3i_2o(x double, y double, z double)
            EMITS (z1 double, z2 double) AS
            class LINE_3I_2O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x") + ctx.getDouble("y"), 3000.0);
                }
            }
            /
        '''))
        
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_boolean(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x BOOLEAN)')
        self.query('insert into fn2.dt values false')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(False,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('BOOLEAN', rows[0][1])

    def test_double(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DOUBLE)')
        self.query('insert into fn2.dt values 32768e100')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(3.2768e+104,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('DOUBLE', rows[0][1])

    def test_dec_32bit(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(9,0))')
        self.query('insert into fn2.dt values 32768')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(32768,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('DECIMAL(9,0)', rows[0][1])

    def test_dec_64bit(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(18,0))')
        self.query('insert into fn2.dt values 32768')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(32768,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('DECIMAL(18,0)', rows[0][1])

    def test_dec_128bit(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(36,0))')
        self.query('insert into fn2.dt values 32768')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(32768,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('DECIMAL(36,0)', rows[0][1])

    def test_dec_32bit_with_scale(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(9,1))')
        self.query('insert into fn2.dt values 99999999.1')
        rows = self.query('''
            select x = 99999999.1 from (select x, fn1.line_1i_1o(0)
            FROM fn2.dt)
            ''')
        self.assertRowsEqual([(True,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('DECIMAL(9,1)', rows[0][1])

    def test_dec_64bit_with_scale(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(18,1))')
        self.query('insert into fn2.dt values 9999999999999999.1')
        rows = self.query('''
            select x = 9999999999999999.1 from (select x, fn1.line_1i_1o(0)
            FROM fn2.dt)
            ''')
        self.assertRowsEqual([(True,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('DECIMAL(18,1)', rows[0][1])

    def test_dec_128bit_with_scale(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(36,1))')
        self.query('insert into fn2.dt values 999999999999999999999999999999999.1')
        rows = self.query('''
            select x = 999999999999999999999999999999999.1 from (select x, fn1.line_1i_1o(0)
            FROM fn2.dt)
            ''')
        self.assertRowsEqual([(True,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('DECIMAL(36,1)', rows[0][1])

    def test_timestamp(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x TIMESTAMP)')
        self.query('''
            insert into fn2.dt values '2010-01-01 23:33:33'
            ''')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(datetime.datetime(2010, 1, 1, 23, 33, 33),0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('TIMESTAMP(3)', rows[0][1])

    def test_timestamp_with_timezone(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x TIMESTAMP WITH LOCAL TIME ZONE)')
        self.query('''
            insert into fn2.dt values '2010-01-01 23:33:33'
            ''')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(datetime.datetime(2010, 1, 1, 23, 33, 33),0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('TIMESTAMP(3) WITH LOCAL TIME ZONE', rows[0][1])

    def test_date(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DATE)')
        self.query('''
            insert into fn2.dt values '2010-01-01'
            ''')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(datetime.date(2010, 1, 1),0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('DATE', rows[0][1])

    def test_varchar_utf8(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x VARCHAR(3000) UTF8)')
        self.query('insert into fn2.dt values repeat(5,300)')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([('5'*300,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('VARCHAR(3000) UTF8', rows[0][1])

    def test_varchar_ascii(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x VARCHAR(3000) ASCII)')
        self.query('insert into fn2.dt values repeat(5,300)')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([('5'*300,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('VARCHAR(3000) ASCII', rows[0][1])

    def test_char_utf8(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x CHAR(2000) UTF8)')
        self.query('insert into fn2.dt values repeat(5,2000)')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([('5'*2000,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('CHAR(2000) UTF8', rows[0][1])

    def test_char_ascii(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x CHAR(2000) ASCII)')
        self.query('insert into fn2.dt values repeat(5,2000)')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([('5'*2000,0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('CHAR(2000) ASCII', rows[0][1])

    def test_interval_ym(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x INTERVAL YEAR TO MONTH)')
        self.query('''
            insert into fn2.dt values '23-11'
            ''')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([('+23-11',0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('INTERVAL YEAR(2) TO MONTH', rows[0][1])

    def test_interval_ds(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x INTERVAL DAY TO SECOND)')
        self.query('''
            insert into fn2.dt values '30 23:33:33'
            ''')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([('+30 23:33:33.000',0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('INTERVAL DAY(2) TO SECOND(3)', rows[0][1])

    def test_geometry(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x GEOMETRY)')
        self.query('''
            insert into fn2.dt values 'POINT(1 1)'
            ''')
        rows = self.query('''
            select x, fn1.line_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([('POINT (1 1)',0,)], rows)
        self.query('create or replace table fn2.foo as select x, fn1.line_1i_1o(0) from fn2.dt')
        rows = self.query('''
            describe fn2.foo
            ''')
        self.assertEqual('GEOMETRY', rows[0][1])


class NullTest(_JavaUdfSetup):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT dob_1i_1o(x double)
            EMITS (y double) AS
            class DOB_1I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"));
                    ctx.emit(ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_1i_1o(x double)
            EMITS (y double) AS
            class LINE_1I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_1i_2o(x double)
            EMITS (y double, z double) AS
            class LINE_1I_2O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x"), ctx.getDouble("x"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_2i_1o(x double, y double)
            EMITS (z double) AS
            class LINE_2I_1O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x") + ctx.getDouble("y"));
                }
            }
            /
        '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT line_3i_2o(x double, y double, z double)
            EMITS (z1 double, z2 double) AS
            class LINE_3I_2O {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(ctx.getDouble("x") + ctx.getDouble("y"), 3000.0);
                }
            }
            /
        '''))
        
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_boolean_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x BOOLEAN)')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0), NVL(x,1)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,True,),(None,0,True,)], rows)

    def test_double_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DOUBLE)')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(x), NVL(x,1)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,None,1,),(None,None,1,)], rows)

    def test_int32_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(9,0))')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0), NVL(x,1)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,1,),(None,0,1,)], rows)

    def test_int64_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(18,0))')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0), NVL(x,1)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,1,),(None,0,1,)], rows)

    def test_int128_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x DECIMAL(36,0))')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0), NVL(x,1)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,1,),(None,0,1,)], rows)

    def test_timestamp_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x timestamp)')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,),(None,0,)], rows)

    def test_date_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x date)')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,),(None,0,)], rows)

    def test_intervalym_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x interval year to month)')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,),(None,0,)], rows)

    def test_intervalds_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x interval day to second)')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,),(None,0,)], rows)

    def test_geo_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x geometry)')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,),(None,0,)], rows)

    def test_varchar_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x varchar(2000))')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,),(None,0,)], rows)

    def test_char_null(self):
        self.query('CREATE OR REPLACE TABLE fn2.DT(x char(2000))')
        self.query('insert into fn2.dt values NULL')
        rows = self.query('''
            select x, fn1.dob_1i_1o(0)
            FROM fn2.dt
            ''')
        self.assertRowsEqual([(None,0,),(None,0,)], rows)



if __name__ == '__main__':
    udf.main()
