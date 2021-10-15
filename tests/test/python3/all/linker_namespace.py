#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../linker-namespace-sanity'))

from exasol_python_test_framework import udf
import linker_namespace_base_test


class LinkerNamespaceTest(linker_namespace_base_test.LinkerNamespaceBaseTest):
    """
    This test verifies that certain libraries are not part of the global linker namespace,
    which is the normal behavior (we create a new linker namespace in the UDF client for those dependencies)
    """

    def test_linker_namespace_udf(self):
        rows = self._execute_linker_namespace_udf(['proto', 'zmq'])
        self.assertGreater(len(rows), 0)
        for item in rows:
            self.assertEqual(None, item[0])


if __name__ == '__main__':
    udf.main()
