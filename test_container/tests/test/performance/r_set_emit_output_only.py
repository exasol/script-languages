#!/usr/bin/env python3
# encoding: utf8

from exasol_python_test_framework import udf
from abstract_performance_test import AbstractPerformanceTest


class SetEmitOutputOnlyRPerformanceTest(AbstractPerformanceTest):

    def setUp(self):
        self.create_schema()
        self.query(udf.fixindent('''
                CREATE R SET SCRIPT OUTPUT_ONLY(input_value INT)
                    EMITS (
                    intVal DECIMAL(9,0), 
                    longVal DECIMAL(18,0), 
                    bigdecimalVal DECIMAL(36,0), 
                    decimalVal DECIMAL(9,2),
                    doubleVal DOUBLE, 
                    doubleIntVal DOUBLE, 
                    stringVal VARCHAR(100), 
                    booleanVal BOOLEAN) AS
                run<-function(ctx){
                    for(i in 1:ctx$input_value){
                        ctx$emit(i,i,i,i,i,i,toString(i),TRUE)
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
