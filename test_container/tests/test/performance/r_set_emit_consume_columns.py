#!/usr/bin/env python3
# encoding: utf8

from exasol_python_test_framework import udf
from abstract_performance_test import AbstractPerformanceTest


class SetEmitConsumeColumnsRPeformanceTest(AbstractPerformanceTest):

    def setUp(self):
        self.create_schema()
        self.generate_data_linear(500)
        self.query(udf.fixindent('''
                CREATE R SET SCRIPT CONSUME_NEXT_COLUMNS(
                    intVal DECIMAL(9,0), 
                    longVal DECIMAL(18,0), 
                    bigdecimalVal DECIMAL(36,0), 
                    decimalVal DECIMAL(9,2),
                    doubleVal DOUBLE, 
                    doubleIntVal DOUBLE, 
                    stringVal VARCHAR(100), 
                    booleanVal BOOLEAN, 
                    dateVal DATE, 
                    timestampVal TIMESTAMP) EMITS (count_value INT) AS
                run <- function(ctx){
                    count <- 0;
                    repeat {
                        if (!ctx$next_row(1))
                            break
                        intVal <- ctx$intVal
                        longVal <- ctx$longVal
                        bigdecimalVal <- ctx$bigdecimalVal
                        decimalVal <- ctx$decimalVal
                        doubleVal <- ctx$doubleVal
                        doubleIntVal <- ctx$doubleIntVal
                        stringVal <- ctx$stringVal
                        booleanVal <- ctx$booleanVal
                        dateVal <- ctx$dateVal
                        timestampVal <- ctx$timestampVal
                        count <- count + 1;
                    }
                    ctx$emit(count);
                }
                '''))
        self.query("commit")
    
    def tearDown(self):
        self.cleanup(self.schema)

    def test_consume_next(self):
        self.run_test(15, 3, 2.0, "SELECT CONSUME_NEXT_COLUMNS(intVal,longVal,bigdecimalVal,decimalVal,doubleVal,doubleIntVal,stringVal,booleanVal,dateVal,timestampVal) FROM T")


if __name__ == '__main__':
    udf.main()
