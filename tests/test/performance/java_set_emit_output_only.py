#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import time

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/..'))

import udf
from abstract_performance_test import AbstractPerformanceTest


class SetEmitOutputOnlyPythonPerformanceTest(AbstractPerformanceTest):

    def setUp(self):
        self.create_schema()
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT OUTPUT_ONLY(input_value INT)
                    EMITS (
                    intVal DECIMAL(9,0), 
                    longVal DECIMAL(18,0), 
                    bigdecimalVal DECIMAL(36,0), 
                    decimalVal DECIMAL(9,2),
                    doubleVal DOUBLE, 
                    doubleIntVal DOUBLE, 
                    stringVal VARCHAR(100), 
                    booleanVal BOOLEAN) AS
                class OUTPUT_ONLY {
                    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        Integer input_value=ctx.getInteger(0);
                        for(int i = 0; i<input_value;i++){
                            ctx.emit(i,i,i,i,i,i,String.valueOf(i),true);
                        }
                    }
                }

                '''))
        self.query("commit")
    
    def tearDown(self):
        self.cleanup(self.schema)

    def test_consume_next(self):
        self.run_test(15, 3, 2.0, "SELECT count(*) FROM (SELECT OUTPUT_ONLY(300000)) as q")

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

