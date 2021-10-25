#!/usr/bin/env python3

from exasol_python_test_framework import udf


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
