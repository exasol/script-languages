#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import time

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/..'))

import udf
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

# vim: ts=4:sts=4:sw=4:et:fdm=indent

