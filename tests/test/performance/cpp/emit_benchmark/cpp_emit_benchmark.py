#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import time

sys.path.append(os.path.realpath(__file__ + '/../../../../../lib'))

import udf


class CppEmitBenchmarkTest(udf.TestCase):

    def setUp(self):
        print("setup")
        self.schema="CppEmitBenchmarkTest"
        self.query("DROP SCHEMA  %s CASCADE" % self.schema, ignore_errors=True)
        self.query("CREATE SCHEMA  %s" % self.schema)
    
    def test_cpp_emit_benchmark(self):
        print("run test")
        self.query(udf.fixindent('''
                CREATE OR REPLACE CPP_EMIT_PERFORMANCE SET SCRIPT %s.emit_only(
                    intVal INT) EMITS (string_value VARCHAR(2000000)) AS
                PLACEHOLDER
                ''' % self.schema))
        rs=self.query("SELECT count(*) FROM (SELECT emit_only(10000000)) as q")
        print(rs)

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

