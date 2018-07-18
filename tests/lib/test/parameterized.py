#!/usr/opt/bs-python-2.7/bin/python

import unittest
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(__file__, '../../../lib')))

from exatest.test import selftest

import exatest
from exatest.testcase import (
        useData,
        ParameterizedTestCase,
        get_sort_key,
        skip,
        skipIf,
        expectedFailure,
        expectedFailureIf,
        )

class ParameterizedTestCaseTest(unittest.TestCase):
    def test_metatest(self):
        class Module:
            class Test(unittest.TestCase):
                def test_pass(self):
                    pass
            
                def test_fail(self):
                    self.fail()

        with selftest(Module) as result:
            self.assertEqual(1, len(result.failures))
            self.assertEqual(2, result.testsRun)
            self.assertIn('test_pass', result.output)
            self.assertIn('test_fail', result.output)
    

    def test_parameterized_tests(self):
        class Module:
            class Test(ParameterizedTestCase):
                data = [(x,) for x in range(10)]
                @useData(data)
                def test_foo(self, x):
                    self.assertTrue(x % 3 != 0)
        with selftest(Module) as result:
            self.assertEqual(10, result.testsRun)
            self.assertEqual(4, len(result.failures))

    def test_large_parameterized_tests(self):
        class Module:
            class Test(ParameterizedTestCase):
                data = [(x,) for x in range(1000)]
                @useData(data)
                def test_foo(self, x):
                    self.assertTrue(x % 3 != 0)
        with selftest(Module) as result:
            self.assertEqual(1000, result.testsRun)
            self.assertEqual(334, len(result.failures))

    def test_parameterized_tests_with_multiple_parameters(self):
        class Module:
            class Test(ParameterizedTestCase):
                @useData([(21, 42)])
                def test_without_docstring(self, x, y):
                    self.assertEquals(21, x)
                    self.assertEquals(42, y)

        with selftest(Module) as result:
            self.assertTrue(result.wasSuccessful())

    def test_parameterized_tests_with_default_docstring(self):
        class Module:
            class Test(ParameterizedTestCase):
                @useData([(21,), (42,)])
                def test_without_docstring(self, x):
                    pass

        with selftest(Module) as result:
            self.assertIn('\ndata: (21,) ... ok', result.output)
            self.assertIn('\ndata: (42,) ... ok', result.output)

    def test_parameterized_tests_with_docstring(self):
        class Module:
            class Test(ParameterizedTestCase):
                @useData([(21,), (42,)])
                def test_with_docstring(self, x):
                    'some_text'
                    pass

        with selftest(Module) as result:
            self.assertIn('\nsome_text; data: (21,) ... ok', result.output)
            self.assertIn('\nsome_text; data: (42,) ... ok', result.output)

class DecoratorInteraction(unittest.TestCase):
    def test_parameterized_tests_with_decorator_skip(self):
        class Module:
            class Test(ParameterizedTestCase):
                @useData((x,) for x in range(2))
                @skip('some reason')
                def test_skipped_with_param(self, x):
                    pass

        with selftest(Module) as result:
            self.assertEqual(2, len(result.skipped))

    def test_parameterized_tests_with_decorator_skipIf(self):
        class Module:
            class Test(ParameterizedTestCase):
                @useData((x,) for x in range(2))
                @skipIf(False, 'Some Reason 1')
                def test_1(self, x): self.assertIn(x, range(2))

                @useData((x,) for x in range(2))
                @skipIf(True, 'Some Reason 2')
                def test_2(self, x): pass

        with selftest(Module) as result:
            self.assertEqual(2, len(result.skipped))
            self.assertNotIn('Some Reason 1', result.output)
            self.assertIn('Some Reason 2', result.output)

    def test_parameterized_tests_with_decorator_expectedFailure(self):
        class Module:
            class Test(ParameterizedTestCase):
                @useData((x,) for x in range(2))
                @expectedFailure
                def test_skipped_with_param(self, x):
                    x/0

        with selftest(Module) as result:
            self.assertEqual(2, len(result.expectedFailures))

class ConditionalTestCaseTest(unittest.TestCase):
    def test_skip(self):
        class Module:
            class Test(ParameterizedTestCase):
                @skip('Some Reason')
                def test_1(self): pass

        with selftest(Module) as result:
            self.assertEqual(1, len(result.skipped))
            self.assertIn('Some Reason', result.output)

    def test_skipIf(self):
        class Module:
            class Test(ParameterizedTestCase):
                @skipIf(False, 'Some Reason 1')
                def test_1(self): pass

                @skipIf(True, 'Some Reason 2')
                def test_2(self): pass

        with selftest(Module) as result:
            self.assertEqual(1, len(result.skipped))
            self.assertNotIn('Some Reason 1', result.output)
            self.assertIn('Some Reason 2', result.output)


    def test_expectedFailure_tests(self):
        class Module:
            class Test(unittest.TestCase):
                def test_1(self): pass
                @expectedFailure
                def test_2(self): self.fail()
                @expectedFailure
                def test_3(self): pass
        
        with selftest(Module) as result:
            self.assertIn('\ntest_2 (__main__.Test) ... expected failure', result.output)
            self.assertIn('\ntest_3 (__main__.Test) ... unexpected success', result.output)
            self.assertEqual(1, len(result.expectedFailures))
            self.assertEqual(1, len(result.unexpectedSuccesses))

    def test_expectedFailureIf_True_tests(self):
        class Module:
            class Test(unittest.TestCase):
                def test_1(self): pass
                @expectedFailureIf(True)
                def test_2(self): self.fail()
                @expectedFailureIf(True)
                def test_3(self): pass
        
        with selftest(Module) as result:
            self.assertIn('\ntest_2 (__main__.Test) ... expected failure', result.output)
            self.assertIn('\ntest_3 (__main__.Test) ... unexpected success', result.output)
            self.assertEqual(1, len(result.expectedFailures))
            self.assertEqual(1, len(result.unexpectedSuccesses))

    def test_expectedFailureIf_False_tests(self):
        class Module:
            class Test(unittest.TestCase):
                def test_1(self): pass
                @expectedFailureIf(False)
                def test_2(self): self.fail()
                @expectedFailureIf(False)
                def test_3(self): pass
        
        with selftest(Module) as result:
            self.assertIn('\ntest_2 (__main__.Test) ... FAIL', result.output)
            self.assertIn('\ntest_3 (__main__.Test) ... ok', result.output)
            self.assertEqual(1, len(result.failures))
            self.assertEqual(0, len(result.expectedFailures))
            self.assertEqual(0, len(result.unexpectedSuccesses))

class TestCaseOrderTest(unittest.TestCase):
    def test_order_of_undecorated_test_methods(self):
        class Module:
            class Test(ParameterizedTestCase):
                def test_03(self): pass
                def test_02(self): pass
                def test_01(self): pass
        self.assertLess(get_sort_key(Module.Test.test_03), get_sort_key(Module.Test.test_02))
        self.assertLess(get_sort_key(Module.Test.test_02), get_sort_key(Module.Test.test_01))

    def test_order_of_skipped_test_methods(self):
        class Module:
            class Test(ParameterizedTestCase):
                def test_03(self): pass
                @skip('broken')
                def test_02(self): pass
                def test_01(self): pass
            class Test1(ParameterizedTestCase):
                def test_03(self): pass
                @skipIf(True, 'broken')
                def test_02(self): pass
                def test_01(self): pass
            class Test2(ParameterizedTestCase):
                def test_03(self): pass
                @skipIf(False, 'broken')
                def test_02(self): pass
                def test_01(self): pass
        self.assertLess(get_sort_key(Module.Test.test_03), get_sort_key(Module.Test.test_02))
        self.assertLess(get_sort_key(Module.Test.test_02), get_sort_key(Module.Test.test_01))
        self.assertLess(get_sort_key(Module.Test1.test_03), get_sort_key(Module.Test1.test_02))
        self.assertLess(get_sort_key(Module.Test1.test_02), get_sort_key(Module.Test1.test_01))
        self.assertLess(get_sort_key(Module.Test2.test_03), get_sort_key(Module.Test2.test_02))
        self.assertLess(get_sort_key(Module.Test2.test_02), get_sort_key(Module.Test2.test_01))

    def test_order_of_expecedFailure_test_methods(self):
        class Module:
            class Test(ParameterizedTestCase):
                def test_03(self): pass
                @expectedFailure
                def test_02(self): pass
                def test_01(self): pass
            class Test1(ParameterizedTestCase):
                def test_03(self): pass
                @expectedFailureIf(True)
                def test_02(self): pass
                def test_01(self): pass
            class Test2(ParameterizedTestCase):
                def test_03(self): pass
                @expectedFailureIf(False)
                def test_02(self): pass
                def test_01(self): pass
        self.assertLess(get_sort_key(Module.Test.test_03), get_sort_key(Module.Test.test_02))
        self.assertLess(get_sort_key(Module.Test.test_02), get_sort_key(Module.Test.test_01))
        self.assertLess(get_sort_key(Module.Test1.test_03), get_sort_key(Module.Test1.test_02))
        self.assertLess(get_sort_key(Module.Test1.test_02), get_sort_key(Module.Test1.test_01))
        self.assertLess(get_sort_key(Module.Test2.test_03), get_sort_key(Module.Test2.test_02))
        self.assertLess(get_sort_key(Module.Test2.test_02), get_sort_key(Module.Test2.test_01))

    def test_order_of_parameterized_test_methods(self):
        class Module:
            class Test(ParameterizedTestCase):
                data = [(x,) for x in range(3)]
                def test_B(self): pass
                @useData(data)
                def test_X(self, x): pass
                def test_A(self): pass
        expected_order = [
                Module.Test.test_B,
                Module.Test.test_X_0,
                Module.Test.test_X_1,
                Module.Test.test_X_2,
                Module.Test.test_A,
                ]
        for i in range(len(expected_order)-1):
            self.assertLess(get_sort_key(expected_order[i]), get_sort_key(expected_order[i+1]))

if __name__ == '__main__':
    unittest.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
