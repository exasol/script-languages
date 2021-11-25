package com.exasol;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.MalformedParameterizedTypeException;
import java.lang.reflect.Method;
import java.lang.reflect.UndeclaredThrowableException;
import java.io.StringWriter;
import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.Arrays;

class ExaStackTraceCleaner {

    public ExaStackTraceCleaner(final String triggerClassName)  {
        mTriggerClassName = triggerClassName;
    }

    public String cleanStackTrace(final Throwable src) {
        final Throwable th = unpack(src);
        cleanExceptionChain(th);
        return format(th);
    }

    private Throwable unpack(final Throwable src) {
        Throwable exc = src;
        while (exc != null && (exc instanceof InvocationTargetException ||
                    exc instanceof MalformedParameterizedTypeException ||
                    exc instanceof UndeclaredThrowableException)) {
            Throwable cause = exc.getCause();
            if (cause == null)
                break;
            else
                exc = cause;
        }
        return exc;
    }

    private String format(final Throwable src) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        src.printStackTrace(pw);
        String stacktrace = sw.toString();
        LinkedList<String> stacktrace_lines = new LinkedList<String>(Arrays.asList(stacktrace.split("\\r?\\n")));

        ListIterator list_Iter = stacktrace_lines.listIterator(0);
        StringWriter stringWriter = new StringWriter();
        PrintWriter writer = new PrintWriter(stringWriter, true);
        while (list_Iter.hasNext()) {
            String line = (String) list_Iter.next();
            writer.println(line.replaceFirst("^\tat ", ""));
        }
        String cleanedStacktrace = stringWriter.toString();
        return cleanedStacktrace;
    }

    private void cleanExceptionChain(final Throwable src) {
        StackTraceElement[] stackTraceElements = src.getStackTrace();
        Integer start_index = null;
        LinkedList<StackTraceElement> newStackTrace = new LinkedList<>();

        if (stackTraceElements.length > 0) {
            for (int idxStackTraceElement = (stackTraceElements.length - 1); idxStackTraceElement >= 0; idxStackTraceElement--) {
                StackTraceElement stackTraceElement = stackTraceElements[idxStackTraceElement];
                boolean addStackTrace = true;
                if (stackTraceElement.getClassName().equals(mTriggerClassName)) {
                    if (start_index == null) {
                        start_index = idxStackTraceElement;
                    }
                } else if ("java.base".equals(stackTraceElement.getModuleName())) {
                    if (start_index != null &&
                            (stackTraceElement.getClassName().startsWith("jdk.internal.reflect") ||
                                    stackTraceElement.getClassName().startsWith("java.lang.reflect"))) {
                        addStackTrace = false;
                    }
                } else {
                    start_index = null;
                }
                if (addStackTrace) {
                    newStackTrace.add(0, stackTraceElement);
                }
            }
            StackTraceElement[] newArr = new StackTraceElement[newStackTrace.size()];
            newArr = newStackTrace.toArray(newArr);
            src.setStackTrace(newArr);
        }
        if (src.getCause() != null) {
            cleanExceptionChain(src.getCause());
        }
    }

    private final String mTriggerClassName;
}