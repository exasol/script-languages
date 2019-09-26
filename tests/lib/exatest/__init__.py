import argparse
import contextlib
import cProfile
import inspect
import logging
import os
import pdb
import pstats
import sys
import tempfile
import time
import unittest
import socket

from unittest import (
        SkipTest,
        suite
        )

import pyodbc

from .threading import Thread
from .testcase import *

@contextlib.contextmanager
def os_timer():
    before = os.times()
    try:
        yield
    finally:
        after = os.times()
        print '\nreal %7.2fs\nuser %7.2fs\nsys  %7.2fs\n' % (
            after[4] - before[4], after[0] - before[0], after[1] - before[1])

@contextlib.contextmanager
def timer():
    class Timer(object):
        def __init__(self):
            self.start = time.time()
            self.stop = None
            self.duration = None
    t = Timer()
    try:
        yield t
    finally:
        t.stop = time.time()
        t.duration = t.stop - t.start

class TestLoader(unittest.TestLoader):
    '''Load tests like the default TestLoader, but sorted by line numbers'''

    def __init__(self, **kwargs):
        self.kwargs=kwargs

    def getTestCaseNames(self, testCaseClass):
        '''Return a sorted sequence of method names found within testCaseClass'''
        def isTestMethod(attrname, testCaseClass=testCaseClass,
                prefix=self.testMethodPrefix):
            return (attrname.startswith(prefix) and
                    hasattr(getattr(testCaseClass, attrname), '__call__'))
        testFnNames = filter(isTestMethod, dir(testCaseClass))
        return sorted(testFnNames, key=lambda x: get_sort_key(getattr(testCaseClass, x)))

    def loadTestsFromModule(self, module, use_load_tests=True):
        '''Return a suite of all tests cases contained in the given module'''
        tests = []
        objects = []
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, TestCase):
                objects.append((inspect.getsourcelines(obj)[1], obj))
        for _, obj in sorted(objects):
            tests.append(self.loadTestsFromTestCase(obj))

        load_tests = getattr(module, 'load_tests', None)
        tests = self.suiteClass(tests)
        if use_load_tests and load_tests is not None:
            try:
                return load_tests(self, tests, None)
            except Exception, e:
                return unittest._make_failed_load_tests(module.__name__, e,
                                               self.suiteClass)
        return tests

    def loadTestsFromTestCase(self, testCaseClass):
        """Return a suite of all tests cases contained in 
           testCaseClass."""
        if issubclass(testCaseClass, suite.TestSuite):
            raise TypeError("Test cases should not be derived from "
                            "TestSuite. Maybe you meant to derive from"
                            " TestCase?")
        testCaseNames = self.getTestCaseNames(testCaseClass)
        if not testCaseNames and hasattr(testCaseClass, 'runTest'):
            testCaseNames = ['runTest']

        # Modification here: parse keyword arguments to testCaseClass.
        test_cases = []
        for test_case_name in testCaseNames:
            test_cases.append(testCaseClass(test_case_name, **self.kwargs))
        loaded_suite = self.suiteClass(test_cases)

        return loaded_suite 

class TestProgram(object):
    logger_name = 'exatest.main'

    def __init__(self):
        self.opts = self._parse_opts()
        self.init_logger()
        self.opts.log = logging.getLogger(self.logger_name)
        self._run()

    def init_logger(self):
        '''initialize logging'''
        datefmt = '%Y-%m-%d %H:%M:%S'
        format = '%(asctime)s.%(msecs)03d '
        #format += os.uname()[1] + ' '
        format += '%(levelname)s '
        format += '[%(name)s] '
        format += '%(message)s'
        formatter = logging.Formatter(format, datefmt)
        logfile = os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.log'
        handler = logging.FileHandler(os.path.join(self.opts.logdir, logfile), mode='w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(self.opts.loglevel)
        console.setFormatter(formatter)

        root = logging.getLogger('')
        root.addHandler(handler)
        root.addHandler(console)
        root.setLevel(logging.DEBUG)

    def _parse_opts(self):
        desc = self.__doc__
        epilog = ''
        parser = argparse.ArgumentParser(description=desc, epilog=epilog)
        parser.add_argument('-v', '--verbose', action='store_const', const=2,
            dest='verbosity', help='verbose output')
        parser.add_argument('-q', '--quiet', action='store_const', const=0,
            dest='verbosity', help='minimal output')
        parser.add_argument('-f', '--failfast', action='store_true',
            help='stop on first failure')

        parser.add_argument('tests', nargs='*',
            help='classes or methods to run in the form "TestCaseClass" or "TestCaseClass.test_method" (default: run all tests)')

        odbc = parser.add_argument_group('ODBC specific')
        odbc.add_argument('--server', help='connection string')
        odbc.add_argument('--user', help='connection user', nargs="?", type=str, default="sys")
        odbc.add_argument('--password', help='connection password', nargs="?", type=str, default="exasol")
        odbc.add_argument('--driver',
            help='path to ODBC driver (default: %(default)s)')
        odbcloglevel = ('off', 'error', 'normal', 'verbose')
        odbc.add_argument('--odbc-log', choices=odbcloglevel,
            help='activate ODBC driver log (default: %(default)s)')

        debug = parser.add_argument_group('generic options')
        choices = ('critical', 'error', 'warning', 'info', 'debug')
        debug.add_argument('--loglevel', choices=choices,
            help='set loglevel for console output (default: %(default)s)')
        debug.add_argument('--debugger', action='store_true',
            help='run program under pdb (Python debugger)')
        debug.add_argument('--profiler', action='store_true',
            help='run program with profiler')
        debug.add_argument('--lint', action='store_true',
            help='static code checker (PyLint)')
        if os.path.exists('/usr/opt/testsystem'):
            driver_path = '/usr/opt/testsystem/EXASolution_ODBC/odbc_sles11_x86_64/lib64/libexaodbc-uo2212.so'
        else:
            driver_path = '/opt/local/testsystem/EXASolution_ODBC/odbc_sles11_x86_64/lib64/libexaodbc-uo2212.so'
        parser.set_defaults(
                verbosity=1,
                failfast=False,
                logdir='.',
                loglevel='warning',
                odbc_log='off',
                connect=None,
                #driver=os.path.realpath(os.path.join(
                #        os.path.abspath(__file__),
                #        '../../../../../lib/libexaodbc-uo2214.so')),
                driver=driver_path,
                debugger=False,
                profiler=False,
                lint=False,
                )
        self.parser_hook(parser)
        opts = parser.parse_args()
        opts.loglevel = getattr(logging, opts.loglevel.upper())
        opts.unittest_args = sys.argv[0:1] + opts.tests
        return opts

    def parser_hook(self, parser):
        '''extend parser in sub classes'''
        pass

    def _run(self):
        with os_timer():
            if self.opts.profiler:
                with tempfile.NamedTemporaryFile() as tmp:
                    cProfile.runctx('self._main()', globals(), locals(), tmp.name)
                    s = pstats.Stats(tmp.name)
                    s.sort_stats("cumulative").print_stats(50)
            elif self.opts.debugger:
                pdb.runcall(self._main)
            elif self.opts.lint:
                self._lint()
            else:
                self._main()

    def prepare_hook(self):
        '''extend preperation in sub classes
        
        Return True if successful.'''
        return True

    def _main(self):
        self.opts.log.info('prepare for tests')
        name = self._write_odbcini()

        os.environ['ODBCINI'] = name
        prepare_ok = self.prepare_hook()
        self.opts.log.info('starting tests')
        testprogram = unittest.main(
                argv=self.opts.unittest_args,
                failfast=self.opts.failfast,
                verbosity=self.opts.verbosity,
                testLoader=TestLoader(dsn="exatest",user=self.opts.user,password=self.opts.password),
                exit=False,
                )
        self.opts.log.info('finished tests')
        rc = 0 if (prepare_ok and
                testprogram.result.wasSuccessful() and
                len(testprogram.result.unexpectedSuccesses) == 0) else 1
        sys.exit(rc)

    def _write_odbcini(self):
        name = os.path.realpath(os.path.join(self.opts.logdir, 'odbc.ini'))
        # we have to resolve the hostname to ipv4 ourselve, because pyodbc sometimes resolve hostnames to ipv6 which is not supported by Exasol
        server=self._resolve_host_to_ipv4(self.opts.server)
        with open(name, 'w') as tmp:
            tmp.write('[ODBC Data Sources]\n')
            tmp.write('exatest=EXASolution\n')
            tmp.write('\n')
            tmp.write('[exatest]\n')
            tmp.write('Driver = %s\n' % self.opts.driver)
            tmp.write('EXAHOST = %s\n' % server)
            tmp.write('EXAUID = %s\n' % self.opts.user)
            tmp.write('EXAPWD = %s\n' % self.opts.password)
            tmp.write('CONNECTIONLCCTYPE = en_US.UTF-8\n')      # TODO Maybe make this optional
            tmp.write('CONNECTIONLCNUMERIC = en_US.UTF-8\n')
            if self.opts.odbc_log != 'off':
                tmp.write('EXALOGFILE = %s/exaodbc.log\n' % self.opts.logdir)
                tmp.write('LOGMODE = %s\n' % {
                        'error': 'ON ERROR ONLY',
                        'normal': 'DEFAULT',
                        'verbose': 'VERBOSE',
                        }[self.opts.odbc_log])
        return name

    def _resolve_host_to_ipv4(self,server):
        host_port_split=server.split(":")
        ip_of_host = socket.gethostbyname(self.opts.server.split(":")[0])
        ip_with_port = ip_of_host+":"+host_port_split[1]
        return ip_with_port

    def _lint(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.realpath(os.path.join(
                os.path.abspath(__file__), '../..'))
        pylint = '/usr/opt/bs-python-2.7/bin/pylint'
        if not os.path.exists(pylint):
            pylint = "pylint"
        cmd = [pylint,
                '--rcfile=%s' % os.path.realpath(
                        os.path.join(os.path.abspath(__file__), '../../pylintrc')),
                sys.argv[0]]
        if os.isatty(sys.stdout.fileno()):
            cmd.append('--output-format=colorized')
        os.execve(cmd[0], cmd, env)

main = TestProgram

# vim: ts=4:sts=4:sw=4:et:fdm=indent
