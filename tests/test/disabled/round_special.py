#!/usr/opt/bs-python-2.7/bin/python

import os
import sys
import decimal

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class Round(udf.TestCase):
    def checkType(self, query, type):
        self.query('''create or replace table tmp as  ''' + query)
        rows = self.query('describe tmp')
        self.assertEqual(rows[0][1], type)


    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('create or replace table t(val double, digits int)')
        self.query('insert into t(digits) values 1,2,3,10,20,30,34,35,36')
        self.query('update t set val = 4/3')

    def test_round_special_double_type(self):
        self.checkType('select round(4/3) a from dual', 'DECIMAL(36,0)')
        self.checkType('select round(4/3, 1) a from dual', 'DECIMAL(36,1)')
        self.checkType('select round(4/3, 2) a from dual', 'DECIMAL(36,2)')
        self.checkType('select round(4/3, 3) a from dual', 'DECIMAL(36,3)')
        self.checkType('select round(4/3, 10) a from dual', 'DECIMAL(36,10)')
        self.checkType('select round(4/3, 20) a from dual', 'DECIMAL(36,20)')
        self.checkType('select round(4/3, 30) a from dual', 'DECIMAL(36,30)')
        self.checkType('select round(4/3, 34) a from dual', 'DECIMAL(36,34)')
        self.checkType('select round(4/3, -1) a from dual', 'DOUBLE')

    def test_round_special_double_type_on_table(self):
        self.checkType('select round(val, 1) a from fn2.t where digits = 1', 'DECIMAL(36,1)')
        self.checkType('select round(val, 2) a from fn2.t where digits = 2', 'DECIMAL(36,2)')
        self.checkType('select round(val, 3) a from fn2.t where digits = 3', 'DECIMAL(36,3)')
        self.checkType('select round(val, 10) a from fn2.t where digits = 10', 'DECIMAL(36,10)')
        self.checkType('select round(val, 20) a from fn2.t where digits = 20', 'DECIMAL(36,20)')
        self.checkType('select round(val, 30) a from fn2.t where digits = 30', 'DECIMAL(36,30)')
        self.checkType('select round(val, 34) a from fn2.t where digits = 34', 'DECIMAL(36,34)')
        self.checkType('select round(val, -1) a from fn2.t where digits = -1', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 1', 'DOUBLE')

    def test_errors(self):
        with self.assertRaisesRegexp(Exception, 'numeric value out of range'):
            self.checkType('select round(4/3, 35) a from dual', 'DOUBLE')
        with self.assertRaisesRegexp(Exception, 'Too many digits in ROUND for castRoundInputToDecimal'):
            self.checkType('select round(4/3, 36) a from dual', 'DOUBLE')
        with self.assertRaisesRegexp(Exception, 'numeric value out of range'):
            self.checkType('select round(val, 35) a from fn2.t where digits = 35', 'DOUBLE')
        with self.assertRaisesRegexp(Exception, 'Too many digits in ROUND for castRoundInputToDecimal'):
            self.checkType('select round(val, 36) a from fn2.t where digits = 36', 'DOUBLE')

    def test_round_double_special_results(self):
        rows = self.query('select round(4/3) a from dual')
        self.assertEqual(rows[0][0], 1)
        rows = self.query('select round(4/3,1) a from dual')
        self.assertEqual(rows[0][0], decimal.Decimal('1.3'))
        rows = self.query('select round(4/3,10) a from dual')
        self.assertEqual(rows[0][0], decimal.Decimal('1.3333333333'))
        rows = self.query('select round(4/3,20) a from dual')
        self.assertEqual(rows[0][0], decimal.Decimal('1.33333333333333324595'))

if __name__ == '__main__':
    udf.main()
