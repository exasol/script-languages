#!/usr/bin/env python2.7

import os
import string
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData
import exatest

class ScriptImport(udf.TestCase):

    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('DROP SCHEMA FN3 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')
        self.query('CREATE SCHEMA FN3')
        self.query('OPEN SCHEMA FN2')

        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            bottom()
            RETURNS INT AS
            class BOTTOM {
                static int f() {
                        return 42;
                    }
                }
            /
            '''))

    def test_ImportError_is_catchable(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                catch_import_exception()
                RETURNS int AS
                class CATCH_IMPORT_EXCEPTION {
                   static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        try {
                            exa.importScript("x");
                        }
                        catch (ExaCompilationException ex) { }
                        return 42;
                    }
                }
                '''))
        rows = self.query('SELECT catch_import_exception() FROM dual')
        self.assertRowsEqual([(42,)], rows)

    def test_preprocessed_ImportError_is_not_catchable(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                cannot_catch_import_exception()
                RETURNS int AS
                %import X;
                class CANNOT_CATCH_IMPORT_EXCEPTION {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'script X not found'):
            rows = self.query('SELECT cannot_catch_import_exception() FROM dual')

    def test_preprocessed_Import_missing_script_name(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                missing_import_script_exception()
                RETURNS int AS
                %import
                '''))
        with self.assertRaisesRegexp(Exception, 'No values found for %import statement'):
            rows = self.query('SELECT missing_import_script_exception() FROM dual')

    def test_preprocessed_Import_missing_script_name2(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                missing_import_script_exception2()
                RETURNS int AS
                %import
                class CANNOT_CATCH_IMPORT_EXCEPTION {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'End of %import statement not found'):
            rows = self.query('SELECT missing_import_script_exception2() FROM dual')

    def test_preprocessed_Import_missing_script_name3(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                missing_import_script_exception3()
                RETURNS int AS
                %import ;
                class CANNOT_CATCH_IMPORT_EXCEPTION {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        X.run(exa, ctx);
                        return 42;
                    }
                }
                '''))
        with self.assertRaisesRegexp(Exception, 'No values found for %import statement'):
            rows = self.query('SELECT missing_import_script_exception3() FROM dual')

    def test_preprocessed_Import_missing_import_end(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                missing_import_end_exception()
                RETURNS int AS
                %import X'''))
        with self.assertRaisesRegexp(Exception, 'End of %import statement not found'):
            rows = self.query('SELECT missing_import_end_exception() FROM dual')

    def test_import_is_case_sensitive(self):
        scripts = [
                ('my_module', 'my_module', 4711),
                ('My_Module', 'My_Module', 42),
                ('MY_MODULE', 'MY_MODULE', 1234),
                ]
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                "%s"()
                RETURNS int AS
                class %s {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return %d;
                    }
                }
                '''
        for triple in scripts:
            self.query(udf.fixindent(sql % triple))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                import_case_sensitive()
                RETURNS int AS
                import java.lang.reflect.Method;
                class IMPORT_CASE_SENSITIVE {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        Class<?> scriptClass = exa.importScript("\\"My_Module\\"");
                        Class[] runParams = {exa.getClass().getInterfaces()[0], ctx.getClass().getInterfaces()[0]};
                        Method runMethod = scriptClass.getDeclaredMethod("run", runParams);
                        Object returnValue = runMethod.invoke(null, exa, ctx);
                        return (returnValue instanceof Integer) ? (int) returnValue : 0;
                    }
                }
                '''))
        rows = self.query('SELECT import_case_sensitive() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_preprocessed_import_is_case_sensitive(self):
        scripts = [
                ('my_module', 'my_module', 4711),
                ('My_Module', 'My_Module', 42),
                ('MY_MODULE', 'MY_MODULE', 1234),
                ]
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                "%s"()
                RETURNS int AS
                class %s {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return %d;
                    }
                }
                '''
        for triple in scripts:
            self.query(udf.fixindent(sql % triple))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                preprocessed_import_case_sensitive()
                RETURNS int AS
                %import "My_Module";
                class PREPROCESSED_IMPORT_CASE_SENSITIVE {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return My_Module.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT preprocessed_import_case_sensitive() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_preprocessed_import_tab(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            X()
            RETURNS int AS
            class X {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 1;
                }
            }
            '''))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                preprocessed_import_case_tab()
                RETURNS int AS
                %import		X;
                class PREPROCESSED_IMPORT_CASE_TAB {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return X.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT preprocessed_import_case_tab() FROM DUAL')
        self.assertRowsEqual([(1,)], rows)

    def test_preprocessed_import_tab_end(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE java SCALAR SCRIPT
            X()
            RETURNS int AS
            class X {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return 1;
                }
            }
            '''))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                preprocessed_import_case_tab_end()
                RETURNS int AS
                %import		X 	 ;
                class PREPROCESSED_IMPORT_CASE_TAB_END {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return X.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT preprocessed_import_case_tab_end() FROM DUAL')
        self.assertRowsEqual([(1,)], rows)

    def test_preprocessed_import_multiple(self):
        scripts = [
                ('A', 'A', 1),
                ('B', 'B', 2),
                ('C', 'C', 3),
                ]
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                "%s"()
                RETURNS int AS
                %%import A;
                %%import B;
                %%import C;
                class %s {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return %d;
                    }
                }
                '''
        for triple in scripts:
            self.query(udf.fixindent(sql % triple))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                X()
                RETURNS int AS
                %import A;
                %import B;
                %import C;
                class X {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return A.run(exa, ctx) + B.run(exa, ctx) + C.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT X() FROM DUAL')
        self.assertRowsEqual([(6,)], rows)

    def test_preprocessed_import_recursive(self):
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                A()
                RETURNS int AS
                %import B;
                class A {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 1 + B.run(exa, ctx);
                    }
                }
                '''
        self.query(udf.fixindent(sql))
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                B()
                RETURNS int AS
                %import C;
                class B {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 2 + C.run(exa, ctx);
                    }
                }
                '''
        self.query(udf.fixindent(sql))
        sql = '''
                CREATE OR REPLACE java SCALAR SCRIPT
                C()
                RETURNS int AS
                class C {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return 3;
                    }
                }
                '''
        self.query(udf.fixindent(sql))

        self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                X()
                RETURNS int AS
                %import A;
                class X {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return A.run(exa, ctx);
                    }
                }
                '''))
        rows = self.query('SELECT X() FROM DUAL')
        self.assertRowsEqual([(6,)], rows)

    def test_import_works(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            %import bottom;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BOTTOM.f();
                }
            }
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_import_is_semi_case_sensitive(self):
        def check(name, classname, n):
            self.query(udf.fixindent('''
                CREATE OR REPLACE java SCALAR SCRIPT
                foo()
                RETURNS INT AS
                %%import %s;
                class FOO {
                    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        return %s.f();
                    }
                }
                /''' % (name, classname)))
            self.assertRowsEqual([(n,)],
                self.query('SELECT foo() FROM DUAL'))

        for name in 'bar', 'Bar', 'BAR':
            self.query(udf.fixindent('''
                CREATE java SCALAR SCRIPT
                "%s"()
                RETURNS INT AS
                class %s {
                    static int f() {
                            return %d;
                    }
                }
                /''' % (name, name, sum(x in string.uppercase for x in name))
                ))

        check("bar", "BAR", 3)
        check("Bar", "BAR", 3)
        check("\"Bar\"", "Bar", 1)
        check("\"bar\"", "bar", 0)

    def test_import_fails_for_lua_script(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            %import bar;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BAR.f();
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE lua SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            function f()
                return 32
            end
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'VM error:.* wrong language LUA'):
            self.query('SELECT foo() FROM DUAL')

    def test_import_fails_for_r_script(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            %import bar;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BAR.f();
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE r SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            f <- function() {
                32
            }
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'VM error:.* wrong language R'):
            self.query('SELECT foo() FROM DUAL')

    def test_import_fails_for_python_script(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            %import bar;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BAR.f();
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE python SCALAR SCRIPT
            bar()
            RETURNS DOUBLE AS

            def f():
                return 32
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'VM error:.* wrong language PYTHON'):
            self.query('SELECT foo() FROM DUAL')

    @useData([
            ('fn2', 'bottom', 'BOTTOM'),
            ('fn2', 'fn2.bottom', 'BOTTOM'),
            ('fn2', 'exa_db.fn2.bottom', 'BOTTOM'),
            ('fn3', 'fn2.bottom', 'BOTTOM'),
            ('fn3', 'exa_db.fn2.bottom', 'BOTTOM')
            ])
    def test_import_works_with_qualified_names(self, schema, name, classname):
        self.query('OPEN SCHEMA %s' % schema)
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            %%import %s;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return %s.f();
                }
            }
            /
            ''' % (name, classname)))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)


    def test_chained_import_works_via_function_call(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            foo()
            RETURNS INT AS
            %import bar;
            class FOO {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BAR.b();
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            bar()
            RETURNS INT AS
            %import bottom;
            class BAR {
                static int b() {
                    return BOTTOM.f();
                }
            }
            /
            '''))
        rows = self.query('SELECT foo() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def test_mutual_import_works(self):
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            ping()
            RETURNS INT AS
            %import pong;
            class PING {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return ping(42);
                }
                static int ping(int n) {
                    if (n > 0)
                        return PONG.pong(n - 1) + 1;
                    else
                        return 0;
                }
            }
            /
            '''))
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            pong()
            RETURNS INT AS
            %import ping;
            class PONG {
                static int pong(int n) {
                    if (n > 0)
                        return PING.ping(n - 1) + 1;
                    else
                        return 0;
                }
            }
            /
            '''))
        rows = self.query('SELECT ping() FROM DUAL')
        self.assertRowsEqual([(42,)], rows)

    def testImportWithView(self):
        self.createUser("foo", "foo")
        self.commit()
        self.query(udf.fixindent('''
            CREATE java SCALAR SCRIPT
            spot42542script()
            RETURNS INT AS
            %import bottom;
            class SPOT42542SCRIPT {
                static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    return BOTTOM.f();
                }
            }
            /
            '''))
        self.query("create or replace view FN2.spot42542_view as SELECT spot42542script() as col FROM DUAL")
        self.query("grant select on FN2.spot42542_view to foo")
        self.commit()
        rows = self.query('''select * from FN2.spot42542_view''')
        self.assertRowsEqual([(42,)], rows)
        foo_conn = self.getConnection('foo', 'foo')
        rows = foo_conn.query('''select * from FN2.spot42542_view''')
        self.assertRowsEqual([(42,)], rows)


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

