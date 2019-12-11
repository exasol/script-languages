#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import time

sys.path.append(os.path.realpath(__file__ + '/../../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/..'))
sys.path.append(os.path.realpath(__file__ + '/../..'))

import udf
from abstract_performance_test import AbstractPerformanceTest

class AbstractScalarEmitConsumeEmitVeryLargePythonPerformanceTest(AbstractPerformanceTest):

    def setup_test(self, python_version="PYTHON"):
        self.create_schema()
        self.generate_data_exponential(4)
        self.query(udf.fixindent('''
                CREATE %s SCALAR SCRIPT CONSUME_COLUMNS(
                        intVal DECIMAL(9,0), 
                        longVal DECIMAL(18,0), 
                        bigdecimalVal DECIMAL(36,0), 
                        decimalVal DECIMAL(9,2),
                        doubleVal DOUBLE, 
                        doubleIntVal DOUBLE, 
                        stringVal VARCHAR(100), 
                        booleanVal BOOLEAN, 
                        dateVal DATE, 
                        timestampVal TIMESTAMP
                    ) EMITS (
                        intVal DECIMAL(9,0), 
                        longVal DECIMAL(18,0), 
                        bigdecimalVal DECIMAL(36,0), 
                        decimalVal DECIMAL(9,2),
                        doubleVal DOUBLE, 
                        doubleIntVal DOUBLE, 
                        stringVal VARCHAR(100), 
                        booleanVal BOOLEAN, 
                        dateVal DATE, 
                        timestampVal TIMESTAMP
                    ) AS
                def run(ctx):
                    intVal = ctx.intVal
                    longVal = ctx.longVal
                    bigdecimalVal = ctx.bigdecimalVal
                    decimalVal = ctx.decimalVal
                    doubleVal = ctx.doubleVal
                    doubleIntVal = ctx.doubleIntVal
                    stringVal = ctx.stringVal
                    booleanVal = ctx.booleanVal
                    dateVal = ctx.dateVal
                    timestampVal = ctx.timestampVal
                    for i in range(10):
                        ctx.emit(intVal,longVal,bigdecimalVal,decimalVal,doubleVal,doubleIntVal,stringVal,booleanVal,dateVal,timestampVal)
                '''%(python_version)))
        self.query("commit")
    
    def execute_consume_next(self):
        self.run_test(1, 0, 2.0, "SELECT count(*) from (SELECT CONSUME_COLUMNS(intVal,longVal,bigdecimalVal,decimalVal,doubleVal,doubleIntVal,stringVal,booleanVal,dateVal,timestampVal) FROM T) as q")

# vim: ts=4:sts=4:sw=4:et:fdm=indent

