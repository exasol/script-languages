package com.exasol;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import com.exasol.ExaStackTraceCleaner;
import java.lang.Exception;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.CoreMatchers.containsString;

//First build some artificial exception chain

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

public class ExaStackTraceCleanerTest {

    private void wrap_custom_test_class() throws ClassNotFoundException, NoSuchMethodException,
                                                    InstantiationException, IllegalAccessException,
                                                    InvocationTargetException {
        Class a = Class.forName("com.exasol.CustomTestClass");
        Constructor<?> constructor = a.getConstructor();
        Object obj = constructor.newInstance();
        Method m = a.getMethod("throwEx");
        m.invoke(obj);
    }

    @Test
    public void runSimpleTest() {
        Throwable th = null;
        try {
            wrap_custom_test_class();
        } catch(final Exception ex) {
           th = ex;
        }
        ExaStackTraceCleaner exaStackTraceCleaner = new ExaStackTraceCleaner();
        final String result = exaStackTraceCleaner.cleanStackTrace(th);
        String[] resultLines = result.split("\n");
        //Check that the
        assertTrue(resultLines.length > 4);
        assertEquals(resultLines[0], "com.exasol.CustomTestException: ex3");
        assertThat(resultLines[1], containsString("com.exasol.CustomTestClass.throwEx(ExaStackTraceCleanerTest.java"));
        assertThat(resultLines[2], containsString("com.exasol.ExaStackTraceCleanerTest.wrap_custom_test_class(ExaStackTraceCleanerTest.java"));
        assertThat(resultLines[3], containsString("com.exasol.ExaStackTraceCleanerTest.runSimpleTest(ExaStackTraceCleanerTest.java"));
    }

}