

class ExaStackTraceCleaner {

  private static void cleanExceptionChain(final Throwable src) {
        StackTraceElement[] stackTraceElements = src.getStackTrace();
        Integer start_index = null;
        LinkedList<StackTraceElement> newStackTrace = new LinkedList<>();

        if (stackTraceElements.length > 0) {
            for (int idxStackTraceElement = (stackTraceElements.length - 1); idxStackTraceElement >= 0; idxStackTraceElement--) {
                StackTraceElement stackTraceElement = stackTraceElements[idxStackTraceElement];
                boolean addStackTrace = true;
                if (stackTraceElement.getClassName().startsWith("com.exasol.thomas.uebensee.testjava.Main")) {
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
}