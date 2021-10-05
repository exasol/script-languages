#!/usr/bin/env python2.7

import os
import sys


sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
import udf


class LinkerNamespaceBaseTest(udf.TestCase):

    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2')

    def _execute_linker_namespace_udf(self, libs_to_check):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + "/../resources/linker_namespace_test_udf.py", "r") as f:
            udf_script = f.read()
        self.query(udf.fixindent('''
                    CREATE OR REPLACE python3 SCALAR SCRIPT linker_namespace_test(search_string VARCHAR(100000)) RETURNS VARCHAR(100000) AS
                    ''' + udf_script))

        lib_value_str = ", ".join("'{0}'".format(lib_str) for lib_str in libs_to_check)
        rows = self.query("SELECT linker_namespace_test(search_string) FROM "
                          "(SELECT search_string FROM (VALUES {0}) AS t(search_string))".format(lib_value_str))
        return rows

