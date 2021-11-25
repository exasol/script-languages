#!/usr/bin/env python3

from exasol_python_test_framework import udf
from pyodbc import DataError

class ExceptionTest(udf.TestCase):

    def setUp(self):
        self.query('CREATE SCHEMA T1', ignore_errors=True)
        self.query('OPEN SCHEMA T1')

    def test_single_call(self):
        exception = None
        self.query(udf.fixindent('''
                create or replace java scalar script
                throw_exception_in_singlecall()
                EMITS (...) AS
                
                class THROW_EXCEPTION_IN_SINGLECALL {
                    public static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        ctx.emit(1);
                    }
                    
                  static String getDefaultOutputColumns(ExaMetadata exa) throws Exception {
                    throw new RuntimeException("Error");
                  }
                }
                '''))

        try:
            self.query("SELECT T1.THROW_EXCEPTION_IN_SINGLECALL() FROM DUAL")
        except DataError as ex:
            exception = ex
        self._verify_exception(exception,
                         "VM error",
                         "com.exasol.ExaUDFException: F-UDF-CL-SL-JAVA-1068: Exception during singleCall getDefaultOutputColumns",
                         "java.lang.RuntimeException: Error",
                         "com.exasol.THROW_EXCEPTION_IN_SINGLECALL.getDefaultOutputColumns",
                         "com.exasol.ExaWrapper.runSingleCall")

    def test_get_integer(self):
        exception = None
        self.query(udf.fixindent('''
                create or replace java scalar script
                error_in_get_integer()
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS
                class ERROR_IN_GET_INTEGER {
                    public static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        ctx.getInteger("ABC");
                    }
                }
                '''))
        try:
            self.query("SELECT T1.error_in_get_integer() FROM DUAL")
        except DataError as ex:
            exception = ex

        self._verify_exception(exception,
                              "VM error",
                              "com.exasol.ExaUDFException: ",
                              "com.exasol.ExaIterationException: E-UDF-CL-SL-JAVA-1123: Column with name 'ABC' does not exist",
                              "com.exasol.ExaIteratorImpl.getInteger",
                              "com.exasol.ERROR_IN_GET_INTEGER.run",
                              "com.exasol.ExaWrapper.run")

    def test_get_connection(self):
        exception = None
        self.query(udf.fixindent('''
                create or replace java scalar script
                error_in_get_connection()
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS
                class ERROR_IN_GET_CONNECTION {
                    public static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        exa.getConnection("ABC");
                    }
                }
                '''))
        try:
            self.query("SELECT T1.error_in_get_connection() FROM DUAL")
        except DataError as ex:
            exception = ex

        self._verify_exception(exception,
                              "VM error",
                              "com.exasol.ExaUDFException: ",
                              "com.exasol.ExaConnectionAccessException: E-UDF-CL-SL-JAVA-1099: connection ABC does not exist",
                              "com.exasol.ExaMetadataImpl.getConnection",
                              "com.exasol.ERROR_IN_GET_CONNECTION.run",
                              "com.exasol.ExaWrapper.run")

    def test_throw_exception(self):
        exception = None
        self.query(udf.fixindent('''
                create or replace java scalar script
                throw_exception()
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS
                class THROW_EXCEPTION {
                    public static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        try {
                            throw new RuntimeException("Error");
                        } catch (final Exception ex) {
                            throw new RuntimeException("Got exception", ex);
                        }
                    }
                }
                '''))

        try:
            self.query("SELECT T1.throw_exception() FROM DUAL")
        except DataError as ex:
            exception = ex
        self._verify_exception(exception,
                               "VM error",
                               "com.exasol.ExaUDFException: F-UDF-CL-SL-JAVA-1080:",
                               "java.lang.RuntimeException: Got exception",
                               "com.exasol.THROW_EXCEPTION.run",
                               "com.exasol.ExaWrapper.run",
                               "Caused by: java.lang.RuntimeException: Error",
                               "com.exasol.THROW_EXCEPTION.run",
                               "... 1 more")

    def test_get_integer(self):
        exception = None
        self.query(udf.fixindent('''
                create or replace java scalar script
                error_in_get_integer()
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS
                class ERROR_IN_GET_INTEGER {
                    public static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        ctx.getInteger("ABC");
                    }
                }
                '''))
        try:
            self.query("SELECT T1.error_in_get_integer() FROM DUAL")
        except DataError as ex:
            exception = ex

        self._verify_exception(exception,
                              "VM error",
                              "com.exasol.ExaUDFException: ",
                              "com.exasol.ExaIterationException: E-UDF-CL-SL-JAVA-1123: Column with name 'ABC' does not exist",
                              "com.exasol.ExaIteratorImpl.getInteger",
                              "com.exasol.ERROR_IN_GET_INTEGER.run",
                              "com.exasol.ExaWrapper.run")

    def test_do_not_filter_user_reflection(self):
        """
            User code reflection code stack traces must not be filtered!
        """
        exception = None
        self.query(udf.fixindent('''
                create or replace java scalar script
                throw_exception_in_user_code_within_reflection()
                EMITS (firstname VARCHAR(100), lastname VARCHAR(100)) AS
                import java.lang.Exception;
                import java.lang.reflect.Constructor;
                import java.lang.reflect.InvocationTargetException;
                import java.lang.reflect.Method;
                final class CustomTestException extends Exception {
                    public CustomTestException(final String msg, final Throwable th) {
                        super(msg, th);
                    }
                }

                final class CustomTestClass {
                    public CustomTestClass() {}
                
                    public void throwEx() throws CustomTestException {
                        try {
                            try {
                                throw new RuntimeException("ex1");
                            } catch (final RuntimeException ex) {
                                throw new RuntimeException("ex2", ex);
                            }
                        } catch(final  RuntimeException ex) {
                            throw new CustomTestException("ex3", ex);
                        }
                    }
                }

                class THROW_EXCEPTION_IN_USER_CODE_WITHIN_REFLECTION {
                        private static void wrap_custom_test_class() throws ClassNotFoundException, 
                                                    NoSuchMethodException,
                                                    InstantiationException, IllegalAccessException,
                                                    InvocationTargetException {
                            Class a = Class.forName("com.exasol.CustomTestClass");
                            Constructor<?> constructor = a.getConstructor();
                            Object obj = constructor.newInstance();
                            Method m = a.getMethod("throwEx");
                            m.invoke(obj);
                        }

                    public static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                        wrap_custom_test_class();
                    }
                }
                '''))
        try:
            self.query("SELECT T1.throw_exception_in_user_code_within_reflection() FROM DUAL")
        except DataError as ex:
            exception = ex

        self._verify_exception(exception,
                              "VM error",
                              "com.exasol.ExaUDFException: ",
                              "com.exasol.CustomTestException: ex3",
                              "com.exasol.CustomTestClass.throwEx(THROW_EXCEPTION_IN_USER_CODE_WITHIN_REFLECTION.java:23)",
                              "java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)",
                              "java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke",
                              "java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke",
                              "java.base/java.lang.reflect.Method.invoke",
                              "com.exasol.THROW_EXCEPTION_IN_USER_CODE_WITHIN_REFLECTION.wrap_custom_test_class",
                              "com.exasol.THROW_EXCEPTION_IN_USER_CODE_WITHIN_REFLECTION.run",
                              "com.exasol.ExaWrapper.run" )



    def _verify_exception(self, exception, *args):
        exception_str = str(exception)
        exception_str_lines = exception_str.split(sep="\\n")
        for idx in range(len(args)):
            if args[idx] not in exception_str_lines[idx]:
                self.fail(f'"{args[idx]}" not in "{exception_str_lines[idx]}"')


if __name__ == '__main__':
    udf.main()
