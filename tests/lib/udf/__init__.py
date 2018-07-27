
import functools
import logging
import os
import re
import sys
import unittest

import pyodbc

import exatest
from exatest import *

from exatest.clients.odbc import ODBCClient

capabilities = []
opts = None

# 
#   TestCase decorators
#
def requires(req):
    '''Skip test if requirements are not met (class method decorator)

    Unlike skipIf decorator, condition is evaluated at method run time.

    Usage:
    class Test(object):

    @requires('foo')
    def test_foo(self):
        # executed

    @requires('bar')
    def test_bar(self):
        # skipped
    '''
    def dec(func):
        @functools.wraps(func)
        def wrapper(*args):
            if not opts.lang:
                raise TypeError('"@requires" is only allowed for generic tests')
            if req not in capabilities:
                raise exatest.SkipTest('requires: %s' % req) 
            return func(*args)
        wrapper._sort_key = exatest.get_sort_key(func)
        return wrapper
    return dec

def expectedFailureIfLang(lang):
    '''Expect test to fail if lang is opts.lang'''
    def dec(func):
        @functools.wraps(func)
        def wrapper(*args):
            if lang == opts.lang:
                return unittest.expectedFailure(func)(*args)
            else:
                return func(*args)
        wrapper._sort_key = exatest.get_sort_key(func)
        return wrapper
    return dec

def fixindent(query):
    '''Remove indent from multi-line query text.

    SQL does not care about indent, but embedded languages (like Python)
    might. Use indent of first indented non-empty line as reference.

    Usage:

        sql = fixindent("""
                CREATE python SCALAR SCRIPT
                foo RETURNS int AS

                def run(context):
                    ...
                """)
    '''
    lines = query.split('\n')
    ref = ''
    indent = re.compile('^(\s+)\S+')
    for line in lines:
        match = indent.match(line)
        if match:
            ref = match.group(1)
            break
    if ref:
        return '\n'.join([
            (line.replace(ref, '', 1) if line.startswith(ref) else line)
            for line in lines])
    else:
        return query

class TestProgram(exatest.TestProgram):
    logger_name = 'udf.main'

    def parser_hook(self, parser):
        new_opts = parser.add_argument_group('UDF specific')
        new_opts.add_argument('--lang',
            help='programming language (default: %(default)s)')
        new_opts.add_argument('--redirector-url',
            help='comma separated list of redirector urls for external script service (default: %(default)s)')
        new_opts.add_argument('--testparam',
            help='comma separated list of parameters for tests (default: %(default)s)')
        new_opts.add_argument('--jdbc-path',
            help='path to the EXASOL JDBC Driver jar file (default: %(default)s)')
        new_opts.add_argument('--is-compat-mode',
            help='Compatibility mode (default: %(default)s)')
        new_opts.add_argument('--script-languages',
            help=' language definition, (default: %(default)s)')
        parser.set_defaults(
                lang=None,
                redirector_url=None,
                param=None,
                jdbc_path=None,
                is_compat_mode=None,
                script_languages=None,
                )

    def prepare_hook(self):
        global opts
        opts = self.opts
        if opts.lang is not None:
            return load_functions(lang=opts.lang, schema='FN1', redirector=opts.redirector_url)
        return True

main = TestProgram

def load_functions(lang=None, schema='FN1', redirector=None):
    client = ODBCClient('exatest')
    client.connect(autocommit=True)
    path = os.path.realpath(os.path.join(os.path.abspath(__file__),
            '../../../lang', lang))
    if not os.path.isdir(path):
        opts.log.critical('%s does not exits', path)
        return False
    _sql(client, 'DROP SCHEMA %s CASCADE' % schema, may_fail=True)
    _sql(client, 'CREATE SCHEMA %s' % schema, fatal_error=True)
    opts.log.info('searching for function definitions beneath %s', path)
    success = True
    for file in _walk(path):
        opts.log.info('loading functions from file %s', file)
        success = _load_file(client, file, redirector) and success
    client.commit()
    capabilities.extend([x[0]
            for x in client.query('''
                SELECT script_name
                    FROM EXA_USER_SCRIPTS
                    WHERE script_schema = ?''', schema)])
    client.close()
    return success

def _sql(client, sql, may_fail=False, fatal_error=False):
    try:
        opts.log.debug('executing SQL: %s', sql)
        client.query(sql)
        return True
    except pyodbc.Error as e:
        if may_fail:
            pass
        else:
            opts.log.critical(str(e))
            opts.log.exception(e)
            if not opts.log.isEnabledFor(logging.DEBUG):
                opts.log.error('SQL was: %s', sql)
            if fatal_error:
                sys.exit(1)
    return False

def _walk(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.sql'):
                yield os.path.join(root, f)

def _load_file(client, path, redirector=None):
    success = True
    for sql in _split_file_on_slash(path):
        sql = _rewrite_redirector(sql, redirector)
        success = _sql(client, sql) and success
    return success

def _rewrite_redirector(sql, redirector):
    if redirector is not None:
        return sql.replace('@@redirector_url@@',
                '\n# redirector '.join(redirector.split(',')))
    else:
        return sql

def _split_file_on_slash(path):
    sql = ''
    for line in open(path):
        line = line.decode('utf8')
        if line == '/\n':
            if sql:
                yield sql
            sql = ''
        else:
            sql += line
    if sql:
        yield sql

class TestCase(exatest.TestCase):
    def query(self, *args, **kwargs):
        new_args = list(args)
        new_args[0] = _rewrite_redirector(new_args[0], opts.redirector_url)
        return super(TestCase, self).query(*new_args, **kwargs)

    

# vim: ts=4:sts=4:sw=4:et:fdm=indent

