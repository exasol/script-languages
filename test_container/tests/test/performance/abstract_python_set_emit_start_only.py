#!/usr/bin/env python3
# encoding: utf8

from exasol_python_test_framework import udf
from abstract_performance_test import AbstractPerformanceTest


class SetEmitStartOnlyPythonPerformanceTest(AbstractPerformanceTest):

    def setup_test(self, python_version="PYTHON"):
        self.create_schema()
        self.query(udf.fixindent('''
                CREATE %s SET SCRIPT START_ONLY(
                    intVal INT) EMITS (count_value INT) AS
                def run(ctx):
                    pass
                ''' % (python_version)))
        self.query("commit")
    
    def execute_start_only(self):
        self.run_test(1000, 3, 2.0, "SELECT START_ONLY(1)")
