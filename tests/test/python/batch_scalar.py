#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

class PythonInterpreter(udf.TestCase):
    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2', ignore_errors=True)

    def test_batch_scalar_returns(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
                batch_scalar_returns(nb int)
                RETURNS int AS

                def run(ctx):
                    while True:
                        ctx.emit(42)
                        if not ctx.next():
                            break
                /
                '''))
        r = self.query("select batch_scalar_returns(21)")
        self.assertEqual(len(r),1)

    def test_batch_scalar_returns_more_emits(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
                batch_scalar_returns(nb int)
                RETURNS int AS

                def run(ctx):
                    while True:
                        ctx.emit(42)
                        ctx.emit(42)
                        if not ctx.next():
                            break
                /
                '''))
        r = self.query("select batch_scalar_returns(21)")
        print(r)
        self.fail()

    def test_batch_scalar_emit(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
                batch_scalar_returns(nb int)
                EMITS (nb int) AS

                def run(ctx):
                    while True:
                        ctx.emit(42)
                        ctx.emit(42)
                        if not ctx.next():
                            break
                /
                '''))
        r = self.query("select batch_scalar_returns(21)")
        self.assertEqual(len(r),2)


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

