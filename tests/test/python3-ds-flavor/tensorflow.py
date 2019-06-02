#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure
from exatest.testcase import skip

class TensorflowBasics(udf.TestCase):
    def setUp(self):
        self.query('create schema tfbasic', ignore_errors=True)

    def test_import_keras(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE python3 scalar SCRIPT tfbasic.import_keras()
                returns varchar(1000) as
                import keras
                import tensorflow
                import tensorflow_hub
                
                def run(ctx):
                    return str(keras.__version__)
                /
                '''))

        row = self.query("select tfbasic.import_keras()")[0]


if __name__ == '__main__':
    udf.main()
