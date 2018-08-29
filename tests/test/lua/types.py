#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class LuaTypes(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_convert_string_to_double(self):
        self.query('''
                CREATE lua SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                function run(ctx)
                    return "1.5"
                end
                ''')
        rows = self.query('''SELECT wrong_type() FROM DUAL''')
        self.assertRowEqual((1.5,), rows[0])

    def test_raises_with_incompatible_type(self):
        self.query('''
                CREATE lua SCALAR SCRIPT
                wrong_type() RETURNS DOUBLE AS
                function run(ctx)
                    return "one point five"
                end
                ''')
        with self.assertRaisesRegexp(Exception, r'bad argument \(number expected, got string\)'):
            self.query('''SELECT wrong_type() FROM DUAL''')

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

