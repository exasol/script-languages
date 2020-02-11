#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

class JavaBugsTest(udf.TestCase):
    def setUp(self):
        self.schema=self.__class__.__name__
        self.query('DROP SCHEMA %s CASCADE'%self.schema, ignore_errors=True)
        self.query('CREATE SCHEMA %s'%self.schema, ignore_errors=True)
        self.query('OPEN SCHEMA %s'%self.schema)

    def test_java_emits_null(self):
        self.query(udf.fixindent('''
        CREATE OR REPLACE JAVA SCALAR SCRIPT JAVA_EMITS() EMITS(i INT) AS
            class JAVA_EMITS {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit(null);
                }
            }
            '''))
        self.query("select JAVA_EMITS();")

    def tearDown(self):
        self.query('DROP SCHEMA %s CASCADE'%self.schema, ignore_errors=True)


if __name__ == '__main__':
    udf.main()

