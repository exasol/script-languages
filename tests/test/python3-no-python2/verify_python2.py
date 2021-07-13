#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf


class NoPython2(udf.TestCase):
    def setUp(self):
        self.query('create schema no_python2', ignore_errors=True)

    def test_no_python2_bin(self):
        got_exception = False
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT no_python2.test_python2_bin_not_available() returns int AS
            import subprocess
            import os.path
            def run(ctx): 
                assert os.path.isdir("/usr/bin/python2") == False
                subprocess.run(["python2", "--version"])
                return 1
            /
            '''))
        try:
            self.query('''SELECT no_python2.test_python2_bin_not_available() FROM dual''')
        except:
            got_exception = True
        assert got_exception

    def test_no_python2_lib(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT no_python2.test_python2_lib_not_available() returns varchar(10000) AS
            import subprocess
            def run(ctx): 
                res = subprocess.run(["find", "/", "-name", "libpython2*.so"], stdout=subprocess.PIPE)
                return res.stdout.decode("utf-8")  #If there are no libraries installed, find should return empty string
            /
            '''))
        try:

            rows = self.query('''SELECT no_python2.test_python2_lib_not_available() FROM dual''')
            print("Res Python2 test:" + str(rows[0]))
            self.assertRowsEqual([('',)], rows)
        except:
            print("Error executing test 'test_no_python2_lib'")
            raise


if __name__ == '__main__':
    udf.main()
