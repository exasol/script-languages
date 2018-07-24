#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../lib'))

import udf

class PyLint(udf.TestCase):

    def test_testmethod_name_used_twice(self):
        pass

    def test_testmethod_name_used_twice(self):
        pass

    def test_undefined_variable(self):
        return foo

class Order2(udf.TestCase):

    @udf.requires('foo')
    def test_30_skipped_1(self):
        pass

    @udf.requires('foo')
    def test_10_skipped_2(self):
        pass

    @udf.skip('foo')
    def test_50_skipped_3(self):
        pass

    @udf.skipIf(True, 'foo')
    def test_40_skipped_4(self):
        pass

    @udf.requires('foo')
    def test_20_skipped_5(self):
        pass

if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
