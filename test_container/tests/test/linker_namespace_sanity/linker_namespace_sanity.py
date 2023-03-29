#!/usr/bin/env python3

import re

from exasol_python_test_framework import udf

from exasol_python_test_framework.exatest.testcase import skipIf
from exasol_python_test_framework.udf import get_supported_languages

from exasol_python_test_framework.exatest.clients.odbc import getScriptLanguagesFromArgs
import linker_namespace_base_test


'''
Purpose of this test if to validate correctness of the other test "linker_namespace.py"
We force protobuf and zmq to be part of the global linkernamespace (by using a specific binary of the UDFclient, created only for this test).
Then we expect that the UDF returns those dependencies. 
'''
class LinkerNamespaceSanityTest(linker_namespace_base_test.LinkerNamespaceBaseTest):

    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2')

    def _setup_language_definition(self):
        lang = getScriptLanguagesFromArgs()
        r_py2 = re.compile(r"exaudf/exaudfclient\b") # Match "...exaudf/exaudfclient ..."/"...exaudf/exaudfclient" but not "...exaudf/exaudfclient_py3"
        lang_static = r_py2.sub("exaudf/exaudfclient_py2_static", lang)
        r_py3 = re.compile(r"exaudf/exaudfclient_py3\b")
        lang_static = r_py3.sub("exaudf/exaudfclient_py3_static", lang_static)
        alter_session_query_str = "ALTER SESSION SET SCRIPT_LANGUAGES='%s'" % lang_static
        print(alter_session_query_str)
        self.query(alter_session_query_str)

    @skipIf('PYTHON3' not in get_supported_languages(), "UDF does not support Python3")
    def test_linker_namespace(self):
        self._setup_language_definition()
        rows = self._execute_linker_namespace_udf(['protobuf', 'zmq'])
        self.assertGreater(len(rows), 0)
        for item in rows:
            self.assertGreater(len(item[0]), 0)


if __name__ == '__main__':
    udf.main()
