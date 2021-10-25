#!/usr/bin/env python3
# encoding: utf8

from exasol_python_test_framework import udf
from abstract_python_scalar_emit_consume_columns import AbstractScalarEmitConsumeColumnsPythonPerformanceTest


class ScalarEmitConsumeColumnsPython3PerformanceTest(AbstractScalarEmitConsumeColumnsPythonPerformanceTest):

    def setUp(self):
        self.setup_test("PYTHON3")

    def tearDown(self):
        self.cleanup(self.schema)

    def test_consume_next(self):
        self.execute_consume_next()


if __name__ == '__main__':
    udf.main()
