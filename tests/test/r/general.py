#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class RInterpreter(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')

    def test_body_is_not_executed_at_creation_time(self):
        self.query(udf.fixindent('''
                CREATE r SCALAR SCRIPT
                body_error()
                RETURNS double AS

                run <--> function(ctx)
                    42
                end
                /
                '''))

    def test_syntax_errors_ignored_at_creation_time(self):
        self.query(udf.fixindent('''
                CREATE lua SCALAR SCRIPT
                syntax_error()
                RETURNS double AS

                run <- function(context)
                    a <--> b
                end
                /
                '''))

if __name__ == '__main__':
    udf.main()
	
# vim: ts=4:sts=4:sw=4:et:fdm=indent

