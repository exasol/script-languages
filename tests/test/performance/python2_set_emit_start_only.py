#!/usr/bin/env python3
# encoding: utf8

from exasol_python_test_framework import udf
from abstract_python_set_emit_start_only import SetEmitStartOnlyPythonPerformanceTest


class SetEmitStartOnlyPython2PerformanceTest(SetEmitStartOnlyPythonPerformanceTest):

    def setUp(self):
        self.setup_test("PYTHON")

    def tearDown(self):
        self.cleanup(self.schema)

    def test_start_only(self):
        self.execute_start_only()


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

