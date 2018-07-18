#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import requires


class Test(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA dynamic_input CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA dynamic_input')
        self.query('CREATE TABLE small(x VARCHAR(2000), y DOUBLE)')
        self.query('''INSERT INTO small VALUES ('Some string ... and some more', 2.2)''')
        self.query('create table groupt(id int, n double, v varchar(999))')
        self.query('''insert into groupt values (1,1,'aa'),
                                                (1,2,'ab'),
                                                (2,2,'ba')
                                                ''')


class DynamicMetadataTest(Test):

    @requires('METADATA_SCALAR_RETURN')
    def test_meta_scalar_return(self):
        rows = self.query('''
            SELECT fn1.metadata_scalar_return('abc', cast(99 as double))
            FROM DUAL
            ''')
        self.assertRowEqual(('2',), rows[0])

    @requires('METADATA_SCALAR_EMIT')
    def test_meta_scalar_emit(self):
        rows = self.query('''
            SELECT fn1.metadata_scalar_emit('abc', cast(99 as double))
            FROM DUAL
            ''')
        self.assertRowEqual(('2',), rows[0])
        self.assertRowEqual(('0',), rows[1])
        self.assertTrue(rows[2][0] == "string" or rows[2][0] == "<type 'unicode'>" or rows[2][0] == "character" or rows[2][0] == "java.lang.String")
        self.assertRowEqual(('CHAR(3) ASCII',), rows[3])
        self.assertRowEqual(('3',), rows[6])
        self.assertRowEqual(('1',), rows[7])
        self.assertTrue(rows[8][0] == 'number' or rows[8][0] == "<type 'float'>" or rows[8][0] == "double" or rows[8][0] == "java.lang.Double")
        self.assertRowEqual(('DOUBLE',), rows[9])


class DynamicInputBasic(Test):
    @requires('BASIC_SCALAR_EMIT')
    def test_basic_scalar_emit_constants(self):
        rows = self.query('''
            SELECT fn1.basic_scalar_emit('abc', cast(99 as double))
            FROM DUAL
            ''')
        self.assertTrue(rows[0][0] == 'abc' or rows[0][0] == "u'abc'")
        self.assertTrue(rows[1][0] == '99' or rows[1][0] == "99.0")

    @requires('BASIC_SCALAR_EMIT')
    def test_basic_scalar_emit(self):
        rows = self.query('''
            SELECT fn1.basic_scalar_emit(x, y)
            FROM small
            ''')
        self.assertTrue(rows[0][0] == 'Some string ... and some more' or rows[0][0] == "u'Some string ... and some more'")
        self.assertRowEqual(('2.2',), rows[1])

    @requires('BASIC_SCALAR_RETURN')
    def test_basic_scalar_return_constants(self):
        rows = self.query('''
            SELECT fn1.basic_scalar_return('abc', cast(99 as double))
            FROM DUAL
            ''')
        self.assertTrue(rows[0][0] == '99' or rows[0][0] == "99.0")

    @requires('BASIC_SCALAR_RETURN')
    def test_basic_scalar_return(self):
        rows = self.query('''
            SELECT fn1.basic_scalar_return(x, y, x, y, x, y, x, y, x, y, x, y, x, y, x, y)
            FROM small
            ''')
        self.assertRowEqual(('2.2',), rows[0])

    @requires('BASIC_SET_EMIT')
    def test_basic_set_emit_constants(self):
        rows = self.query('''
            SELECT fn1.basic_set_emit(cast(99 as double),'77','aaaa')
            FROM DUAL
            ''')
        self.assertTrue(rows[0][0] == '99' or rows[0][0] == "99.0")
        self.assertTrue(rows[1][0] == '77' or rows[1][0] == "u'77'")
        self.assertTrue(rows[2][0] == 'aaaa' or rows[2][0] == "u'aaaa'")
        self.assertTrue(rows[3][0] == 'result:  , 99 , 77 , aaaa' or rows[3][0] == "result: 99.0 , u'77' , u'aaaa' , " or rows[3][0] == "result: 99 , 77 , aaaa , " or rows[3][0] == "result: 99.0 , 77 , aaaa , ")

    @requires('BASIC_SET_EMIT')
    def test_basic_set_emit(self):
        rows = self.query('''
            SELECT fn1.basic_set_emit(n, v)
            FROM groupt GROUP BY id ORDER BY 1
            ''')
        self.assertTrue(rows[0][0] == '1' or rows[0][0] == "1.0")
        self.assertTrue(rows[1][0] == '2' or rows[1][0] == "2.0")
        self.assertTrue(rows[2][0] == '2' or rows[2][0] == "2.0")
        self.assertTrue(rows[3][0] == 'aa' or rows[3][0] == "result: 1.0 , u'aa' , 2.0 , u'ab' , ")
        self.assertTrue(rows[4][0] == 'ab' or rows[4][0] == "result: 2.0 , u'ba' , ")
        self.assertTrue(rows[5][0] == 'ba' or rows[5][0] == "u'aa'")
        self.assertTrue(rows[6][0] == 'result:  , 1 , aa , 2 , ab' or rows[6][0] == "u'ab'" or rows[6][0] == "result: 1 , aa , 2 , ab , " or rows[6][0] == "result: 1.0 , aa , 2.0 , ab , ")
        self.assertTrue(rows[7][0] == 'result:  , 2 , ba' or rows[7][0] == "u'ba'" or rows[7][0] == "result: 2 , ba , " or rows[7][0] == "result: 2.0 , ba , ")

    @requires('BASIC_SET_EMIT')
    def test_basic_set_emit_one_group(self):
        rows = self.query('''
            SELECT fn1.basic_set_emit(cast(id as double), n, v)
            FROM groupt ORDER BY 1
            ''')
        self.assertTrue(rows[0][0] == '1' or rows[0][0] == "1.0")
        self.assertTrue(rows[1][0] == '1' or rows[1][0] == "1.0")
        self.assertTrue(rows[2][0] == '1' or rows[2][0] == "1.0")
        self.assertTrue(rows[3][0] == '2' or rows[3][0] == "2.0")
        self.assertTrue(rows[4][0] == '2' or rows[4][0] == "2.0")
        self.assertTrue(rows[5][0] == '2' or rows[5][0] == "2.0")
        self.assertTrue(rows[6][0] == 'aa' or rows[7][0] == "u'aa'")
        self.assertTrue(rows[7][0] == 'ab' or rows[8][0] == "u'ab'")
        self.assertTrue(rows[8][0] == 'ba' or rows[9][0] == "u'ba'")
        self.assertTrue(rows[9][0] == 'result:  , 1 , 1 , aa , 2 , 2 , ba , 1 , 2 , ab' \
                            or rows[6][0] == "result: 1.0 , 1.0 , u'aa' , 2.0 , 2.0 , u'ba' , 1.0 , 2.0 , u'ab' , " \
                            or rows[9][0] == "result: 1 , 1 , aa , 2 , 2 , ba , 1 , 2 , ab , " \
                            or rows[9][0] == "result: 1.0 , 1.0 , aa , 2.0 , 2.0 , ba , 1.0 , 2.0 , ab , ")

    @requires('BASIC_SET_RETURN')
    def test_basic_set_return_constants(self):
        rows = self.query('''
            SELECT fn1.basic_set_return(cast(99 as double),'77','aaaa')
            FROM DUAL
            ''')
        self.assertTrue(rows[0][0] == 'result:  , 99  , 77  , aaaa ' \
                            or rows[0][0] == "result: 99.0 , u'77' , u'aaaa' , " \
                            or rows[0][0] == "result: 99 , 77 , aaaa , " \
                            or rows[0][0] == "result: 99.0 , 77 , aaaa , ")

    @requires('BASIC_SET_RETURN')
    def test_basic_set_return(self):
        rows = self.query('''
            SELECT fn1.basic_set_return(n, v)
            FROM groupt GROUP BY id ORDER BY 1
            ''')
        self.assertTrue(rows[0][0] == 'result:  , 1  , aa  , 2  , ab ' \
                            or rows[0][0] == "result: 1.0 , u'aa' , 2.0 , u'ab' , " \
                            or rows[0][0] == "result: 1 , aa , 2 , ab , " \
                            or rows[0][0] == "result: 1.0 , aa , 2.0 , ab , ")
        self.assertTrue(rows[1][0] == 'result:  , 2  , ba ' \
                            or rows[1][0] == "result: 2.0 , u'ba' , " \
                            or rows[1][0] == "result: 2 , ba , " \
                            or rows [1][0] == "result: 2.0 , ba , ")

    @requires('BASIC_SET_RETURN')
    def test_basic_set_return_one_group(self):
        rows = self.query('''
            SELECT fn1.basic_set_return(cast(id as double), n, v)
            FROM groupt
            ''')
        self.assertTrue(rows[0][0] == 'result:  , 1  , 1  , aa  , 2  , 2  , ba  , 1  , 2  , ab ' \
                            or rows[0][0] == "result: 1.0 , 1.0 , u'aa' , 2.0 , 2.0 , u'ba' , 1.0 , 2.0 , u'ab' , " \
                            or rows[0][0] == "result: 1 , 1 , aa , 2 , 2 , ba , 1 , 2 , ab , " \
                            or rows[0][0] == "result: 1.0 , 1.0 , aa , 2.0 , 2.0 , ba , 1.0 , 2.0 , ab , ")


class DynamicInputDatatypeSpecific(Test):
    @requires('TYPE_SPECIFIC_ADD')
    def test_type_specific_add_string(self):
        rows = self.query('''
            SELECT fn1.type_specific_add(v, v, v)
            FROM groupt
            ''')
        self.assertTrue('result:  , aa , aa , aa , ba , ba , ba , ab , ab , ab' == rows[0][0] or 'result: aa , aa , aa , ba , ba , ba , ab , ab , ab , ' == rows[0][0])

    @requires('TYPE_SPECIFIC_ADD')
    def test_type_specific_add_number(self):
        rows = self.query('''
            SELECT fn1.type_specific_add(n,n,n,n,n,n,n,n,n,n)
            FROM groupt
            ''')
        self.assertTrue(rows[0][0] == 'result:  50' or rows[0][0] == "result: 50.0" or rows[0][0] == 'result: 50')


class DynamicInputErrors(Test):
    @requires('WRONG_ARG')
    def test_exception_wrong_arg(self):
        if udf.opts.lang == 'r':
            raise udf.SkipTest('does not work with R currently')
        err_text = {
            'lua': 'out of range',
            'python': 'does not exist',
            'ext-python': 'does not exist',
            'java': 'does not exist',
            }
        with self.assertRaisesRegexp(Exception, err_text[udf.opts.lang]):
            self.query('''select fn1.wrong_arg('a') from dual''')

    @requires('WRONG_OPERATION')
    def test_exception_wrong_operation(self):
        err_text = {
            'lua': 'attempt to perform arithmetic on field',
            'r': 'non-numeric argument to binary operator',
            'python': 'multiply sequence by non-int of type',
            'ext-python': 'multiply sequence by non-int of type',
            'java': 'bad operand types for binary operator',
            }
        with self.assertRaisesRegexp(Exception, err_text[udf.opts.lang]):
            self.query('''select fn1.wrong_operation('a','b') from dual''')

    @requires('EMPTY_SET_RETURNS')
    def test_exception_empty_set_returns(self):
        with self.assertRaisesRegexp(Exception, 'user defined set script has no arguments'):
            self.query('''select fn1.empty_set_returns() from groupt''')

    @requires('EMPTY_SET_EMITS')
    def test_exception_empty_set_emits(self):
        with self.assertRaisesRegexp(Exception, 'user defined set script has no arguments'):
            self.query('''select fn1.empty_set_emits() from groupt''')

class DynamicInputOptimizations(Test):
    @requires('BASIC_SCALAR_EMIT')
    @requires('BASIC_SET_RETURN')
    def test_mapreduce_optimization(self):
        rows = self.query('''
            select fn1.basic_set_return("v") from ( select fn1.basic_scalar_emit(n,n,n,n,n,n,n,n,n,n) from groupt)
            ''')
        self.assertTrue(rows[0][0] == 'result:  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 1  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2  , 2 ' \
                            or rows[0][0] == "result: u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'1.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , u'2.0' , " \
                            or rows[0][0] == "result: 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , " \
                            or rows[0][0] == "result: 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , 2.0 , ")


if __name__ == '__main__':
    udf.main()
