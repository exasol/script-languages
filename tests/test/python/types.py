#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class PythonTypes(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_not_convert_string_to_double(self):
        self.query(udf.fixindent('''
                CREATE python SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                def run(ctx):
                    return "one point five"
                '''))
        with self.assertRaisesRegexp(Exception, r'type float but data given have type'):
            self.query('''SELECT wrong_type() FROM DUAL''')

    def test_raises_with_incompatible_type(self):
        self.query(udf.fixindent('''
                CREATE python SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                def run(ctx):
                    return "one point five"
                '''))
        with self.assertRaisesRegexp(Exception, r'type float but data given have type'):
            self.query('''SELECT wrong_type() FROM DUAL''')


if __name__ == '__main__':
    udf.main()
                
# vim: ts=4:sts=4:sw=4:et:fdm=indent
