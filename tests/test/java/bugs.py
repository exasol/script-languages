#!/usr/bin/env python3

from exasol_python_test_framework import udf


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

