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

class AbstractScalarEmitOutputOnlyVeryLargePythonPerformanceTest(AbstractPerformanceTest):

    def setup_test(self, python_version="PYTHON"):
        self.create_schema()
        self.generate_data_linear(1,base=1)
        self.query(udf.fixindent('''
                CREATE {} SCALAR SCRIPT OUTPUT_ONLY(
                        input_value DECIMAL(18,0),
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
                    )
                    EMITS (
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

                from datetime import timedelta
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

                    for i in range(ctx.input_value):
                        add_value = i%10
                        ctx.emit(
                            intVal+add_value,
                            longVal+add_value,
                            bigdecimalVal+add_value,
                            decimalVal+add_value,
                            doubleVal+add_value,
                            doubleIntVal+add_value,
                            stringVal+str(add_value),
                            booleanVal and bool(i%2),
                            dateVal+timedelta(days=add_value),
                            timestampVal+timedelta(days=add_value))
                '''.format(python_version)))
        self.query("commit")
    
    def execute_consume_next(self):
        emit_size=1000*10**4
        self.run_test(1, 0, 2.0, "SELECT count(*) from (SELECT OUTPUT_ONLY(%s, intVal,longVal,bigdecimalVal,decimalVal,doubleVal,doubleIntVal,stringVal,booleanVal,dateVal,timestampVal) FROM T) as q" % emit_size)

# vim: ts=4:sts=4:sw=4:et:fdm=indent

