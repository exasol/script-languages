#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf


class NoPython2(udf.TestCase):
    def setUp(self):
        self.query('create schema no_python2', ignore_errors=True)

    def test_no_python2_bin(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT no_python2.test_python2_bin_not_available() returns int AS
            import subprocess
            import os.path
            def run(ctx): 
                retVal = 1 if os.path.isdir("/usr/bin/python2") else 0
                try:
                    subprocess.run(["python2", "--version"])
                    retVal += 1
                except:
                    pass
                    
                return retVal
            /
            '''))
        try:
            rows = self.query('''SELECT no_python2.test_python2_bin_not_available() FROM dual''')
            self.assertRowsEqual([(0,)], rows)
        except:
            print("Error executing test 'test_no_python2_bin'")
            raise

    def test_no_python2_lib(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT no_python2.test_python2_lib_not_available() returns varchar(10000) AS
            import subprocess
            def run(ctx): 
                res = subprocess.run(["find", "/", "-name", "libpython2*.so", 
                    "!", "-path", "/buckets/*", 
                    "!", "-path", "/tmp/*"], stdout=subprocess.PIPE)
                return res.stdout.decode("utf-8")  #If there are no libraries installed, find should return empty string
            /
            '''))
        try:
            rows = self.query('''SELECT no_python2.test_python2_lib_not_available() FROM dual''')
            self.assertRowsEqual([(None,)], rows)
        except:
            print("Error executing test 'test_no_python2_lib'")
            raise


if __name__ == '__main__':
    udf.main()
