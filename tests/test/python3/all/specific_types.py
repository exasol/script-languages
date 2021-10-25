#!/usr/bin/env python3
# encoding: utf8

from exasol_python_test_framework import udf


class PythonTypes(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_not_convert_string_to_double(self):
        self.query(udf.fixindent('''
                CREATE python3 SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                def run(ctx):
                    return "one point five"
                '''))
        with self.assertRaisesRegex(Exception, r'type float but data given have type'):
            self.query('''SELECT wrong_type() FROM DUAL''')

    def test_raises_with_incompatible_type(self):
        self.query(udf.fixindent('''
                CREATE python3 SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                def run(ctx):
                    return "one point five"
                '''))
        with self.assertRaisesRegex(Exception, r'type float but data given have type'):
            self.query('''SELECT wrong_type() FROM DUAL''')


if __name__ == '__main__':
    udf.main()
