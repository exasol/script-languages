#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

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

    def test_round_double_type(self):
        self.checkType('select round(4/3) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 1) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 2) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 3) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 10) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 20) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 30) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 34) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 35) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, 36) a from dual', 'DOUBLE')
        self.checkType('select round(4/3, -1) a from dual', 'DOUBLE')

    def test_round_double_type_on_table(self):
        self.checkType('select round(val, digits) a from fn2.t where digits = 1', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 2', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 3', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 10', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 20', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 30', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 34', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 35', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = 36', 'DOUBLE')
        self.checkType('select round(val, digits) a from fn2.t where digits = -1', 'DOUBLE')
    

if __name__ == '__main__':
    udf.main()
