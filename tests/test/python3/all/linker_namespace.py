#!/usr/bin/env python2.7

import os
import sys


sys.path.append(os.path.realpath(__file__ + '/../../../../lib'))
sys.path.append(os.path.realpath(__file__ + '/../../../linker-namespace-sanity'))

import udf
import linker_namespace_base_test

class LinkerNamespaceTest(linker_namespace_base_test.LinkerNamespaceBaseTest):

    def test_linker_namespace_udf(self):
        rows = self._execute_linker_namespace_udf(['proto', 'zmq'])
        self.assertGreater(len(rows), 0)
        for item in rows:
            self.assertEqual(None, item[0])


if __name__ == '__main__':
    udf.main()
