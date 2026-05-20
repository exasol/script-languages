#!/usr/bin/env python3

from exasol_python_test_framework import udf



class _Python3UdfSetup(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT
            basic_scalar_emit( ... )
            EMITS ("v" VARCHAR(2000)) as
            def run(ctx):
                i = 0
                while i < exa.meta.input_column_count:
                    ctx.emit(repr(ctx[i]))
                    i = i + 1
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT
            basic_scalar_return( ... )
            RETURNS VARCHAR(2000) AS
            def run(ctx):
                return repr(ctx[exa.meta.input_column_count-1])
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT
            basic_set_emit( ... )
            EMITS ("v" VARCHAR(2000)) AS
            def run(ctx):
                    var = 'result: '
                    while True:
                            for i in range (0,exa.meta.input_column_count):
                                    ctx.emit(repr(ctx[i]))
                                    var = var + repr(ctx[i]) + ' , '
                            if not ctx.next(): break
                    ctx.emit(var)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT
            basic_set_return( ... )
            RETURNS VARCHAR(2000) AS
            def run(ctx):
                    var = 'result: '
                    while True:
                            for i in range (0,exa.meta.input_column_count):
                                    var = var + repr(ctx[i]) + ' , '
                            if not ctx.next(): break
                    return var
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT
            empty_set_emits( ... )
            EMITS (x varchar(2000)) AS
            def run(ctx):
                return 1
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT
            empty_set_returns( ... )
            RETURNS varchar(2000) AS
            def run(ctx):
                return 1
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT
            metadata_scalar_emit (...)
            EMITS("v" VARCHAR(2000)) AS
            def run(ctx):
                ctx.emit(repr(exa.meta.input_column_count))
                for i in range (0,exa.meta.input_column_count):
                    ctx.emit(exa.meta.input_columns[i].name)
                    ctx.emit(repr(exa.meta.input_columns[i].type))
                    ctx.emit(exa.meta.input_columns[i].sql_type)
                    ctx.emit(repr(exa.meta.input_columns[i].precision))
                    ctx.emit(repr(exa.meta.input_columns[i].scale))
                    ctx.emit(repr(exa.meta.input_columns[i].length))
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT
            metadata_scalar_return (...)
            RETURNS VARCHAR(2000) AS
            def run(ctx):
                return repr(exa.meta.input_column_count)
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SET SCRIPT
            type_specific_add(...)
            RETURNS VARCHAR(2000) AS
            def run(ctx):
                    var = 'result: '
                    if repr(exa.meta.input_columns[0].type) == "<type 'unicode'>" or repr(exa.meta.input_columns[0].type) == "<class 'str'>":
                            while True:
                                    for i in range (0,exa.meta.input_column_count):
                                            var = var + ctx[i] + ' , '
                                    if not ctx.next(): break
                    else:
                            sum = 0
                            while True:
                                    for i in range (0,exa.meta.input_column_count):
                                            sum = sum + ctx[i]
                                    if not ctx.next(): break
                            var = var + repr(sum)
                    return var
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT
            wrong_arg(...)
            RETURNS varchar(2000) AS
            def run(ctx):
                return ctx[1]
            /
        '''))

        self.query(udf.fixindent('''
            CREATE PYTHON3 SCALAR SCRIPT
            wrong_operation(...)
            RETURNS varchar(2000) AS
            def run(ctx):
                return ctx[0] * ctx[1]
            /
        '''))
        
        self.query('DROP SCHEMA dynamic_input CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA dynamic_input')
        self.query('CREATE TABLE dynamic_input.small(x VARCHAR(2000), y DOUBLE)')
        self.query('''INSERT INTO dynamic_input.small VALUES ('Some string ... and some more', 2.2)''')
        self.query('create table dynamic_input.groupt(id int, n double, v varchar(999))')
        self.query('''insert into dynamic_input.groupt values (1,1,'aa'),
                                                (1,2,'ab'),
                                                (2,2,'ba')
                                                ''')

class Test(_Python3UdfSetup):
    pass


class DynamicMetadataTest(Test):

    def test_meta_scalar_return(self):
        rows = self.query('''
            SELECT fn1.metadata_scalar_return('abc', cast(99 as double))
            FROM DUAL
            ''')
        self.assertRowEqual(('2',), rows[0])

    def test_meta_scalar_emit(self):
        rows = self.query('''
            SELECT fn1.metadata_scalar_emit('abc', cast(99 as double))
            FROM DUAL
            ''')
        self.assertRowEqual(('2',), rows[0])
        self.assertRowEqual(('0',), rows[1])
        self.assertTrue(rows[2][0] == "string" or rows[2][0] == "<type 'unicode'>" or rows[2][0] == "character" or rows[2][0] == "java.lang.String" or rows[2][0] == "<class 'str'>")
        self.assertRowEqual(('CHAR(3) ASCII',), rows[3])
        self.assertRowEqual(('3',), rows[6])
        self.assertRowEqual(('1',), rows[7])
        self.assertTrue(rows[8][0] == 'number' or rows[8][0] == "<type 'float'>" or rows[8][0] == "double" or rows[8][0] == "java.lang.Double" or rows[8][0] == "<class 'float'>")
        self.assertRowEqual(('DOUBLE',), rows[9])


class DynamicInputBasic(Test):
    def test_basic_scalar_emit_constants(self):
        rows = self.query('''
            SELECT fn1.basic_scalar_emit('abc', cast(99 as double))
            FROM DUAL
            ''')
        self.assertTrue(rows[0][0] == 'abc' or rows[0][0] == "u'abc'" or rows[0][0] == "'abc'")
        self.assertTrue(rows[1][0] == '99' or rows[1][0] == "99.0")

    def test_basic_scalar_emit(self):
        rows = self.query('''
            SELECT fn1.basic_scalar_emit(x, y)
            FROM dynamic_input.small
            ''')
        self.assertTrue(rows[0][0] == 'Some string ... and some more' or rows[0][0] == "u'Some string ... and some more'" or rows[0][0] == "'Some string ... and some more'")
        self.assertRowEqual(('2.2',), rows[1])

    def test_basic_scalar_return_constants(self):
        rows = self.query('''
            SELECT fn1.basic_scalar_return('abc', cast(99 as double))
            FROM DUAL
            ''')
        self.assertTrue(rows[0][0] == '99' or rows[0][0] == "99.0")

    def test_basic_scalar_return(self):
        rows = self.query('''
            SELECT fn1.basic_scalar_return(x, y, x, y, x, y, x, y, x, y, x, y, x, y, x, y)
            FROM dynamic_input.small
            ''')
        self.assertRowEqual(('2.2',), rows[0])

    def test_basic_set_emit_constants(self):
        rows = self.query('''
            SELECT fn1.basic_set_emit(cast(99 as double),'77','aaaa')
            FROM DUAL
            ''')
        print("0---:"+str(rows[3][0]))
        self.assertTrue(rows[0][0] == '99' or rows[0][0] == "99.0")
        self.assertTrue(rows[1][0] == '77' or rows[1][0] == "u'77'" or rows[1][0] == "'77'")
        self.assertTrue(rows[2][0] == 'aaaa' or rows[2][0] == "u'aaaa'" or rows[2][0] == "'aaaa'")
        self.assertTrue(rows[3][0] == 'result:  , 99 , 77 , aaaa' or rows[3][0] == "result: 99.0 , u'77' , u'aaaa' , " or rows[3][0] == "result: 99 , 77 , aaaa , " or rows[3][0] == "result: 99.0 , 77 , aaaa , " or rows[3][0] == "result: 99.0 , '77' , 'aaaa' , ")

    def test_basic_set_emit(self):
        rows = self.query('''
            SELECT fn1.basic_set_emit(n, v)
            FROM groupt GROUP BY id ORDER BY 1
            ''')
        self.assertTrue(rows[0][0] == '1' or rows[0][0] == "1.0" or rows[0][0] == "'aa'")
        self.assertTrue(rows[1][0] == '2' or rows[1][0] == "2.0" or rows[1][0] == "'ab'")
        self.assertTrue(rows[2][0] == '2' or rows[2][0] == "2.0" or rows[2][0] == "'ba'")
        self.assertTrue(rows[3][0] == 'aa' or rows[3][0] == "result: 1.0 , u'aa' , 2.0 , u'ab' , " or rows[3][0] == "1.0")
        self.assertTrue(rows[4][0] == 'ab' or rows[4][0] == "result: 2.0 , u'ba' , "  or rows[4][0] == "2.0")
        self.assertTrue(rows[5][0] == 'ba' or rows[5][0] == "u'aa'" or rows[5][0] == "2.0")
        self.assertTrue(rows[6][0] == 'result:  , 1 , aa , 2 , ab' or rows[6][0] == "u'ab'" or rows[6][0] == "result: 1 , aa , 2 , ab , " or rows[6][0] == "result: 1.0 , aa , 2.0 , ab , "  or rows[6][0] == "result: 1.0 , 'aa' , 2.0 , 'ab' , ")
        self.assertTrue(rows[7][0] == 'result:  , 2 , ba' or rows[7][0] == "u'ba'" or rows[7][0] == "result: 2 , ba , " or rows[7][0] == "result: 2.0 , ba , " or rows[7][0] == "result: 2.0 , 'ba' , ")

    def test_basic_set_emit_one_group(self):
        rows = self.query('''
            SELECT fn1.basic_set_emit(cast(id as double), n, v)
            FROM groupt ORDER BY 1
            ''')
        self.assertTrue(rows[0][0] == '1' or rows[0][0] == "1.0" or rows[0][0] == "'aa'")
        self.assertTrue(rows[1][0] == '1' or rows[1][0] == "1.0" or rows[1][0] == "'ab'")
        self.assertTrue(rows[2][0] == '1' or rows[2][0] == "1.0" or rows[2][0] == "'ba'")
        self.assertTrue(rows[3][0] == '2' or rows[3][0] == "2.0" or rows[3][0] == "1.0")
        self.assertTrue(rows[4][0] == '2' or rows[4][0] == "2.0" or rows[4][0] == "1.0")
        self.assertTrue(rows[5][0] == '2' or rows[5][0] == "2.0" or rows[5][0] == "1.0")
        self.assertTrue(rows[6][0] == 'aa' or rows[7][0] == "u'aa'" or rows[6][0] == "2.0")
        self.assertTrue(rows[7][0] == 'ab' or rows[8][0] == "u'ab'" or rows[7][0] == "2.0")
        self.assertTrue(rows[8][0] == 'ba' or rows[9][0] == "u'ba'" or rows[8][0] == "2.0")
        self.assertTrue(rows[9][0] == 'result:  , 1 , 1 , aa , 2 , 2 , ba , 1 , 2 , ab' \
                            or rows[6][0] == "result: 1.0 , 1.0 , u'aa' , 2.0 , 2.0 , u'ba' , 1.0 , 2.0 , u'ab' , " \
                            or rows[9][0] == "result: 1 , 1 , aa , 2 , 2 , ba , 1 , 2 , ab , " \
                            or rows[9][0] == "result: 1.0 , 1.0 , aa , 2.0 , 2.0 , ba , 1.0 , 2.0 , ab , " or rows[9][0] == "result: 1.0 , 1.0 , 'aa' , 2.0 , 2.0 , 'ba' , 1.0 , 2.0 , 'ab' , ")

    def test_basic_set_return_constants(self):
        rows = self.query('''
            SELECT fn1.basic_set_return(cast(99 as double),'77','aaaa')
            FROM DUAL
            ''')
        self.assertTrue(rows[0][0] == 'result:  , 99  , 77  , aaaa ' \
                            or rows[0][0] == "result: 99.0 , u'77' , u'aaaa' , " \
                            or rows[0][0] == "result: 99 , 77 , aaaa , " \
                            or rows[0][0] == "result: 99.0 , 77 , aaaa , " or rows[0][0] == "result: 99.0 , '77' , 'aaaa' , ")

    def test_basic_set_return(self):
        rows = self.query('''
            SELECT fn1.basic_set_return(n, v)
            FROM groupt GROUP BY id ORDER BY 1
            ''')
        self.assertTrue(rows[0][0] == 'result:  , 1  , aa  , 2  , ab ' \
                            or rows[0][0] == "result: 1.0 , u'aa' , 2.0 , u'ab' , " \
                            or rows[0][0] == "result: 1 , aa , 2 , ab , " \
                            or rows[0][0] == "result: 1.0 , aa , 2.0 , ab , " or rows[0][0] == "result: 1.0 , 'aa' , 2.0 , 'ab' , ")
        self.assertTrue(rows[1][0] == 'result:  , 2  , ba ' \
                            or rows[1][0] == "result: 2.0 , u'ba' , " \
                            or rows[1][0] == "result: 2 , ba , " \
                            or rows [1][0] == "result: 2.0 , ba , " or rows [1][0] == "result: 2.0 , 'ba' , ")

    def test_basic_set_return_one_group(self):
        rows = self.query('''
            SELECT fn1.basic_set_return(cast(id as double), n, v)
            FROM groupt
            ''')
        self.assertTrue(rows[0][0] == 'result:  , 1  , 1  , aa  , 2  , 2  , ba  , 1  , 2  , ab ' \
                            or rows[0][0] == "result: 1.0 , 1.0 , u'aa' , 2.0 , 2.0 , u'ba' , 1.0 , 2.0 , u'ab' , " \
                            or rows[0][0] == "result: 1 , 1 , aa , 2 , 2 , ba , 1 , 2 , ab , " \
                            or rows[0][0] == "result: 1.0 , 1.0 , aa , 2.0 , 2.0 , ba , 1.0 , 2.0 , ab , " or rows[0][0] == "result: 1.0 , 1.0 , 'aa' , 2.0 , 2.0 , 'ba' , 1.0 , 2.0 , 'ab' , ")


class DynamicInputDatatypeSpecific(Test):
    def test_type_specific_add_string(self):
        rows = self.query('''
            SELECT fn1.type_specific_add(v, v, v)
            FROM groupt
            ''')
        self.assertTrue('result:  , aa , aa , aa , ba , ba , ba , ab , ab , ab' == rows[0][0] or 'result: aa , aa , aa , ba , ba , ba , ab , ab , ab , ' == rows[0][0])

    def test_type_specific_add_number(self):
        rows = self.query('''
            SELECT fn1.type_specific_add(n,n,n,n,n,n,n,n,n,n)
            FROM groupt
            ''')
        self.assertTrue(rows[0][0] == 'result:  50' or rows[0][0] == "result: 50.0" or rows[0][0] == 'result: 50')


class DynamicInputErrors(Test):
    def test_exception_wrong_arg(self):
        if udf.opts.lang == 'r':
            raise udf.SkipTest('does not work with R currently')
        err_text = {
            'lua': 'out of range',
            'python3': 'does not exist',
            'java': 'does not exist',
            }
        with self.assertRaisesRegex(Exception, err_text[udf.opts.lang]):
            self.query('''select fn1.wrong_arg('a') from dual''')

    def test_exception_wrong_operation(self):
        err_text = {
            'lua': 'attempt to perform arithmetic on field',
            'r': 'non-numeric argument to binary operator',
            'python3': 'multiply sequence by non-int of type',
            'java': 'bad operand types for binary operator',
            }
        with self.assertRaisesRegex(Exception, err_text[udf.opts.lang]):
            self.query('''select fn1.wrong_operation('a','b') from dual''')

    def test_exception_empty_set_returns(self):
        with self.assertRaisesRegex(Exception, 'data exception - missing input parameters for SET UDF script'):
            self.query('''select fn1.empty_set_returns() from dynamic_input.groupt''')

    def test_exception_empty_set_emits(self):
        with self.assertRaisesRegex(Exception, 'data exception - missing input parameters for SET UDF script'):
            self.query('''select fn1.empty_set_emits() from dynamic_input.groupt''')

class DynamicInputOptimizations(Test):
    def test_mapreduce_optimization(self):
        rows = self.query('''
            select fn1.basic_set_return("v") from ( select fn1.basic_scalar_emit(n,n,n,n,n,n,n,n,n,n) from dynamic_input.groupt)
            ''')
        self.assertTrue(rows[0][0] == 'result:  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2 ' \
                            or rows[0][0] == "result: u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , " \
                            or rows[0][0] == "result: 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , " \
                            or rows[0][0] == "result: 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , " or rows[0][0] == "result: '1.0' , '1.0' , '1.0' , '1.0' , '1.0' , '1.0' , '1.0' , '1.0' , '1.0' , '1.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , '2.0' , ")


if __name__ == '__main__':
    udf.main()
