#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import time

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/..'))

import udf
from abstract_python_scalar_emit_output_only_very_large import AbstractScalarEmitOutputOnlyVeryLargePythonPerformanceTest


class ScalarEmitOutputOnlyVeryLargePython2PerformanceTest(AbstractScalarEmitOutputOnlyVeryLargePythonPerformanceTest):

    def setUp(self):
        self.setup_test("PYTHON")

    def tearDown(self):
        self.cleanup(self.schema)

    def test_consume_next(self):
        self.execute_consume_next()


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

