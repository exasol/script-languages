#!/usr/bin/env python2.7

import os
import sys
import re

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
import udf
from exatest.clients.odbc import getScriptLanguagesFromArgs
import linker_namespace_base_test

class LinkerNamespaceSanityTest(linker_namespace_base_test.LinkerNamespaceBaseTest):

    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2')

    def test_linker_namespace(self):

        lang = getScriptLanguagesFromArgs()
        r_py2 = re.compile(r"exaudf/exaudfclient\b") # Match "...exaudf/exaudfclient ..."/"...exaudf/exaudfclient" but not "...exaudf/exaudfclient_py3"
        lang_static = r_py2.sub("exaudf/exaudfclient_py2_static", lang)
        r_py3 = re.compile(r"exaudf/exaudfclient_py3\b")
        lang_static = r_py3.sub("exaudf/exaudfclient_py3_static", lang_static)
        alter_session_query_str = "ALTER SESSION SET SCRIPT_LANGUAGES='%s'" % lang_static
        print(alter_session_query_str)
        self.query(alter_session_query_str)
        rows = self._execute_linker_namespace_udf(['protobuf', 'zmq'])
        self.assertGreater(len(rows), 0)
        for item in rows:
            self.assertGreater(len(item[0]), 0)

if __name__ == '__main__':
    udf.main()
