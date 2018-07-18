#!/usr/opt/bs-python-2.7/bin/python
# coding: UTF-8
'''
This is an annotated test file of the UDF testing framework. You can
execute it directly.
The only mandatory option is --server.

The framework is based on Python's unittest framework, so Python's 
documentation applies here as well.
'''


import os
import sys

# Every test file must import udf; the "sys.path.append" line is used
# to tell Pythen there to find udf:
sys.path.append(os.path.realpath(__file__ + '/../../lib'))

import udf

# Global setup and cleanup can be put in the following two functions:
def setUpModule():
    pass

def tearDownModule():
    pass

# Feel free to define helper functions at module, class or instance level
def my_helper(foo):
    return foo + 1


# The framework executes every method named "test*" in classes
# derived from udf.TestCase. Every test-method gets its own instance.
# To pass data between tests, your have to use class variables.

class StandardPythonUnitTestCase(udf.TestCase):
    '''If you like, write class level documentation as a multi-line
    string like this.


    Consult Python's documentation for the unittest module for details.'''

    @classmethod
    def setUpClass(cls):
        '''This method is executed only once before the first test in this
        class.'''

        # set a class-level variable:
        cls.foo = 42

    @classmethod
    def tearDownClass(cls):
        '''executed after the last test'''
        pass

    def setUp(self):
        '''executed before each test'''
        pass

    def tearDown(self):
        '''executed after each test'''
        pass

    def some_helper_function(self):
        '''not a test because it's name does not start with "test"'''
        pass

    def test_some_test_case(self):
        self.assertTrue(4, 2+2)
        self.assertLessEqual(2, 3)
        self.assertIn(4, [1,2,3,4])
        self.assertIsNotNone(42)
    
    def test_catched_exceptions(self):
        with self.assertRaises(IndexError):
            a = []
            print a[10]

        # check for exception texts:
        with self.assertRaisesRegexp(Exception, 'FooBar.*Error'):
            raise ValueError('FooBarBazError')

    def test_case_with_comment(self):
        '''Test case can have a comment

        The first line is printed with --verbose
        '''
        pass

class ConditionalTestExecution(udf.TestCase):

    @udf.skip("This test is skipped")
    def test_some_skipped_test(self):
        1/0

    @udf.skipIf(0!=2, "wrong math")
    def test_modulus(self):
        self.assertEqual(0, 1+1)   

    @udf.expectedFailure
    def test_expected_explosion(self):
        1/0



#
# The following example are specific to udf.TestCase:
#

class ParameterizedTests(udf.TestCase):

    data1 = [
            (1,),
            (2,),
            ]

    @udf.useData(data1)
    def test_parameterized_test_with_one_parameter(self, p):
        self.assertIn(p, [1, 2])

    @udf.useData((a, a+1) for a in range(4))
    def test_parameterized_test_with_two_parameters(self, x, y):
        self.assertEqual(x, y-1)



class ConditionalTestExecution(udf.TestCase):

    @udf.expectedFailureIfLang('very_exotic_language')
    def test_case1(self):
        pass

    @udf.requires('some_Function')
    def test_not_defined_function(self):
        '''Not executed if "some_Function" is not defined for this language'''
        1/0

    
    @udf.requires('some_Function')
    @udf.requires('some_other_Function')
    def test_requirements_and_skipIfs_can_be_stacked(self):
        1/0

    def test_skipping_without_decorators(self):
        if udf.opts.lang != 'very_exotic_language':
            raise udf.SkipTest('only works with very_exotic_language')

class CombinationsOfDecorators(udf.TestCase):
    
    @udf.expectedFailureIfLang('very_exotic_language')
    @udf.requires('SOME_FUNCTION')
    def test_two_decorators(self):
        1/0

    @udf.useData((x,) for x in range(2))
    @udf.requires('some_function')
    def test_parameterization_works_with_other_decorators_1(self, x):
        '''useData must be the first decorator'''
        self.assertIn(x, range(10))

    @udf.useData((x,) for x in range(2))
    @udf.expectedFailureIfLang('very_exotic_language')
    def test_parameterization_works_with_other_decorators_2(self, x):
        '''useData must be the first decorator'''
        self.assertIn(x, range(10))

class Queries(udf.TestCase):
    def test_sql_hello_world(self):
        rows = self.query('SELECT * FROM DUAL')
        self.assertEqual(1, self.rowcount(), "expected 1 row")
        self.assertEqual(1, len(rows), "same")
        
        self.assertNotEqual([(None,)], rows, "rows are lists of 'row' objects")

        self.assertRowsEqual([(None,)], rows)



class Unicode(udf.TestCase):
    '''Unicode and ODBC have a somewhat fuzzy relationship...
        
    Python: Unicode strings starts with u: u'B\xe4r', u'\uf98a', u'\U00010380'
   
            If the second line of the script is "# coding: UTF-8", unicode literals
            can be included directly: u'Bär', u'の', u'力'

            For printing, file writing etc. Unicode strings must be converted to UTF-8 or
            something else.

            u'Bär'.encode('utf8') ---> 'B\xc3\xa4r' (binary data string)
           '\xe3\x81\xae'.decode('utf8') ---> u'\u306e'

            The codecs module provides a wrapper for file operations.

            Full (UTF-32) Unicode is supported if sys.maxunicode == 1114111

    
    
    '''

    @udf.expectedFailure
    def test_sending_query_text_as_unicode(self):
        rows = self.query(
            # This is a standard Python string substitution, so self.query() sees only
            # a single string:
            u'''
            SELECT 42
            FROM DUAL
            WHERE UNICODECHR(382) = %s''' % unichr(382))
        self.assertEqual(42, rows[0][0])

    def test_sending_unicode_as_prepared_statement(self):
        rows = self.query(
            '''
            SELECT 42
            FROM DUAL
            WHERE UNICODECHR(382) = ?''', unichr(382))
        self.assertEqual(42, rows[0][0])

    def test_receiving_unicode_data_somehow_works_for_result_sets(self):
        '''This tests emits a UnicodeWarning because of the failed comparison'''
        rows = self.query('SELECT UNICODECHR(382) FROM DUAL')
        # For some reasons, in this combination of unixODBC/pyODBC/EXAODBC,
        # Unicode is yielded as UTF-8 encoded binary data, not as a Python
        # unicode string!
        self.assertNotEqual(unichr(382), rows[0][0])

        # Decoding by hand works somehow, but this may not work with the next
        # version of pyODBC/unixODBC/EXAODBC. Do not count on it!
        self.assertEqual(unichr(382), rows[0][0].decode('utf8'))


# The next to lines tell Python to run the udf.main method only if this
# test is executed as a program (and not included as a module): 
if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
