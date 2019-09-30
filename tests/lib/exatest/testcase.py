
import functools
import inspect
import itertools
import logging
import pyodbc
import sys
import traceback
import unittest

from contextlib import contextmanager

from .clients.odbc import ODBCClient

#
# extend/replace decorator (add sort_key_feature)
#

def skip(reason):
    def decorator(test_item):
        wrapped = unittest.skip(reason)(test_item)
        wrapped._sort_key = get_sort_key(test_item)
        return wrapped
    return decorator

def skipIf(condition, reason):
    """
    Skip a test if the condition is true.
    """
    if condition:
        return skip(reason)
    return lambda x: x

def skipUnless(condition, reason):
    """
    Skip a test unless the condition is true.
    """
    if not condition:
        return skip(reason)
    return lambda x: x

def expectedFailure(func):
    wrapped = unittest.expectedFailure(func)
    wrapped._sort_key = get_sort_key(func)
    return wrapped

def expectedFailureIf(condition):
    if condition:
        return expectedFailure
    else:
        return lambda x: x

def get_sort_key(obj, name=None):
    '''Helper function to sort test methods by line number, not by name'''
    if hasattr(obj, '_sort_key'):
        return getattr(obj, '_sort_key')
    else:
        fname = name if name is not None else obj.__name__
        return (inspect.getsourcelines(obj)[1], fname)

#
# Parameterized tests
#
# Usage:
#   class Test(ParameterizedTestCase):
#
#       data = [(1,), (2,), (3,)]
#
#       @useData(data)
#       def test_with_parameter(self, x):
#           # do something with x
#

class ParameterizedTest(list):
    pass

def useData(generator):
    data = list(generator)
    name_format = '%%s_%%0%dd' % len(str(len(data) - 1))
    def decorator(f):
        tests = ParameterizedTest()
        def closure(i, args):
            @functools.wraps(f)
            def wrapper(self):
                return f(self, *args)
            wrapper.__name__ = name_format % (f.__name__, i)
            doc = '%s; ' % f.__doc__ if f.__doc__ else ''
            wrapper.__doc__ = '%sdata: %s' % (doc, repr(args))
            wrapper._sort_key = get_sort_key(f, wrapper.__name__)
            return wrapper
        for i, args in enumerate(data):
            tests.append(closure(i, args))
        return tests
    return decorator

class _ParameterizedTestCaseMetaclass(type):
    def __new__(cls, name, bases, dct):
        new_tests = {}
        for obj in dct.values():
            if isinstance(obj, ParameterizedTest):
                new_tests.update({test.__name__: test for test in obj})
        dct.update(new_tests)
        new_cls = super(_ParameterizedTestCaseMetaclass, cls).__new__(
                cls, name, bases, dct)
        new_cls.log = logging.getLogger(name)
        return new_cls

class ParameterizedTestCase(unittest.TestCase):
    __metaclass__ = _ParameterizedTestCaseMetaclass

#
# TestCase with integrated database connection
#

class _DBConnectionTestCaseMetaclass(_ParameterizedTestCaseMetaclass):
    def __new__(cls, name, bases, dct):
        new_cls = super(_DBConnectionTestCaseMetaclass, cls).__new__(
                cls, name, bases, dct)
        if 'setUp' in dct:
            new_cls._setUp = dct['setUp']
        if 'tearDown' in dct:
            new_cls._tearDown = dct['tearDown']

        new_cls.setUp = new_cls._TestCase__setUpWrapper
        new_cls.tearDown = new_cls._TestCase__tearDownWrapper
        return new_cls

class TestCase(ParameterizedTestCase):
    __metaclass__ = _DBConnectionTestCaseMetaclass

    def __init__(self, *args, **kwargs):
        self.dsn = kwargs["dsn"]
        self.user = kwargs["user"]
        self.password= kwargs["password"]
        # We need to remove dsn, user and password before we forward the kwargs, because the base class doesn't expect these parameters and throws an error
        del kwargs["dsn"]
        del kwargs["user"]
        del kwargs["password"]
        super(TestCase, self).__init__(*args, **kwargs)
        self._expectations = []
        self._needs_assertExpectations = False


    def __setUpWrapper(self):
        self.__preSetUp()
        try:
            self._setUp()
        except:
            self.__postTearDown()
            raise

    def __tearDownWrapper(self):
        self._tearDown()
        self.__postTearDown()

    def __preSetUp(self):
        self.log = logging.getLogger(
                '%s.%s' % (self.__class__.__name__, self._testMethodName))
        self._client = ODBCClient(self.dsn,self.user,self.password)
        self.log.debug('connecting to DSN %s with User %s'%(self.dsn,self.user))
        try:
            self._client.connect()
        except Exception as e:
            self.log.critical(str(e))
            raise
    
    def _setUp(self):
        pass
    
    def _tearDown(self):
        pass

    def __postTearDown(self):
        self.log.info('disconnection from DSN %s'% self.dsn)
        self._client.close()
        if self._needs_assertExpectations:
            self.assertExpectations()
            self.fail("expectTrue() etc. used, but not followed by assertExpectations()")

    def query(self, *args, **kwargs):
        self.log.debug('executing SQL: %s', args[0])
        try:
            return self._client.query(*args)
        except Exception as e:
            if not kwargs.get('ignore_errors'):
                self.log.error('executing SQL failed: %s: %s', e.__class__.__name__, e)
                if not self.log.isEnabledFor(logging.DEBUG):
                    self.log.error('executed SQL was: %s', args[0])
                raise

    def executeStatement(self, *args, **kwargs):
        self.log.debug('executing SQL: %s', args[0])
        try:
            return self._client.executeStatement(*args)
        except Exception as e:
            if not kwargs.get('ignore_errors'):
                self.log.error('executing SQL failed: %s: %s', e.__class__.__name__, e)
                if not self.log.isEnabledFor(logging.DEBUG):
                    self.log.error('executed SQL was: %s', args[0])
                raise

    def queryScalar(self, *args, **kwargs):
        rows = self.query(*args, **kwargs)
        self.assertEqual(len(rows), 1, msg='Scalar query result should have 1 row, but has %s'%len(rows))
        self.assertEqual(len(rows[0]), 1, msg='Scalar query result should have 1 column, but has %s'%len(rows[0]))
        return rows[0][0]

    def rowcount(self):
        return self._client.rowcount()
        
    def columnNames(self):
        return [colDescription[0] for colDescription in self._client.cursorDescription()]
        
    def cursorDescription(self):
        return self._client.cursorDescription()

    def commit(self):
        self._client.commit()

    def rollback(self):
        self._client.rollback()

    def assertRowEqual(self, left, right, msg=None):
        lrow = tuple(left)
        rrow = tuple(right)
        self.assertEqual(lrow, rrow, msg)
        
    def assertRowsEqualIgnoreOrder(self, left, right, msg=None):
        lrows = [tuple(x) for x in left]
        rrows = [tuple(x) for x in right]
        self.assertEqual(sorted(lrows), sorted(rrows), msg)

    def assertRowsEqual(self, left, right, msg=None):
        # TODO Proposal: Only convert to tuples if necessary, i.e. if not already a list of tuples (or only if this is a list of pyodbc.rows)
        lrows = [tuple(x) for x in left]
        rrows = [tuple(x) for x in right]
        self.assertEqual(lrows, rrows, msg)

    def expectRowEqual(self, left, right, msg=None):
        lrow = tuple(left)
        rrow = tuple(right)
        self.expectEqual(lrow, rrow, msg)

    def expectRowsEqualIgnoreOrder(self, left, right, msg=None):
        lrows = [tuple(x) for x in left]
        rrows = [tuple(x) for x in right]
        self.expectEqual(sorted(lrows), sorted(rrows), msg)

    def expectRowsEqual(self, left, right, msg=None):
        # TODO Proposal: Only convert to tuples if necessary, i.e. if not already a list of tuples (or only if this is a list of pyodbc.rows)
        lrows = [tuple(x) for x in left]
        rrows = [tuple(x) for x in right]
        self.expectEqual(lrows, rrows, msg)

    def getConnection(self, username, password):
        client = ODBCClient(self.dsn,self.user,self.password)
        self.log.debug('connecting to {dsn} for user {username}'.format(dsn=self.dsn, username=username))
        client.connect(uid = username, pwd = password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username = username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username = username, password = password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

    #
    # Expectations work like assertions, but are not immediately fatal.
    #
    # Every assertion (even self-defined) can be turned in an expectation.
    #
    # Usage:
    #
    #   def assertFoo(self, bar):
    #       if something(bar):
    #           self.fail("Not foo enough")
    #
    #   def test_something(self):
    #       with self.expectations():
    #           self.expectTrue(1 != 1)
    #           self.expectEquals(21, 42)
    #           self.expectFoo(42)
    #       self.assertFoo(something_else)
    #
    # Unfulfilled expectations result in an AssertionError after leaving the with context
    # Alternatively, expectations can be checked with self.assertExpectations().
    # Unchecked expectations result in a test error, even if the expectations are fulfilled.
    #
    # Style: If you write a test with lots of expectations, especially if they are not checked
    # immediatly, think about splitting your test. It is probably too long and tries to test too
    # many different things.

    def assertExpectations(self):
        self._needs_assertExpectations = False
        if self._expectations:
            info = "\n".join(self._expectations)
            self._expectations = []
            self.fail("unfulfilled expectations:\n\n" + info)
   
    @contextmanager
    def expectations(self):
        yield
        self.assertExpectations() 

    def __getattr__(self, name):
        if name.startswith('expect'):
            # replace expectFoo with assertFoo
            newname = name.replace('expect', 'assert', 1)
            try:
                newattr = getattr(self, newname)
                def wrapper(*args, **kwargs):
                    self._needs_assertExpectations = True
                    try:
                        newattr(*args, **kwargs)
                    except self.failureException as e:
                        stack = traceback.extract_stack()
                        stack = reversed(stack)
                        stack = itertools.takewhile(lambda x: "testMethod()" not in x, stack)
                        stack = itertools.islice(stack, 1, None)
                        stack = reversed(list(stack))
                        
                        tb = 'Traceback (most recent call last):\n'
                        tb += ''.join(traceback.format_list(stack))
                        # replace AssertionError with ExpectationError
                        tb += 'ExpectationError: ' + str(e) + '\n'
                        self._expectations.append(tb)
                return wrapper
            except AttributeError:
                # do not raise for newname, but for name instead
                pass
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    #try:
    #except self.failureException as e:

class ExpectationError(TestCase.failureException):
    pass

# vim: ts=4:sts=4:sw=4:et:fdm=indent
