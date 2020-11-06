#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

class GetpassTest(udf.TestCase):
    def setUp(self):
        self.query('CREATE SCHEMA getpass', ignore_errors=True)
        self.query('OPEN SCHEMA getpass', ignore_errors=True)

    def tearDown(self):
        self.query('DROP SCHEMA getpass CASCADE', ignore_errors=True)

    def test_getuser(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE python SCALAR SCRIPT
                get_user_from_passwd()
                RETURNS VARCHAR(10000) AS

                def run(ctx):
                    import getpass
                    return getpass.getuser()
                /
                '''))
        rows = self.query("select get_user_from_passwd()")
        expected = u"exadefusr"
        self.assertEqual(expected ,rows[0][0])


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

