#!/usr/bin/env python3
# encoding: utf8


from exasol_python_test_framework import udf
from abstract_performance_test import AbstractPerformanceTest


class SetEmitStartOnlyRPeformanceTest(AbstractPerformanceTest):

    def setUp(self):
        self.create_schema()
        self.generate_data_linear(500)
        self.query(udf.fixindent('''
                CREATE R SET SCRIPT START_ONLY(
                    intVal INT) EMITS (count_value INT) AS
                run <- function(ctx){
                }
                '''))
        self.query("commit")
    
    def tearDown(self):
        self.cleanup(self.schema)

    def test_consume_next(self):
        self.run_test(1000, 3, 2.0, "SELECT START_ONLY(1)")


if __name__ == '__main__':
    udf.main()
