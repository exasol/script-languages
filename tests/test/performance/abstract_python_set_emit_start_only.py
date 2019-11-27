#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import time

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/..'))

import udf
from abstract_performance_test import AbstractPerformanceTest


class SetEmitStartOnlyPythonPerformanceTest(AbstractPerformanceTest):

    def setup_test(self, python_version="PYTHON"):
        self.create_schema()
        self.query(udf.fixindent('''
                CREATE %s SET SCRIPT START_ONLY(
                    intVal INT) EMITS (count_value INT) AS
                def run(ctx):
                    pass
                '''%(python_version)))
        self.query("commit")
    
    def execute_start_only(self):
        self.run_test(1000, 2.0, "SELECT START_ONLY(1)")

# vim: ts=4:sts=4:sw=4:et:fdm=indent

