package com.exasol;

import java.io.IOError;
import java.io.StringWriter;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.MalformedParameterizedTypeException;
import java.lang.reflect.Method;
import java.lang.reflect.UndeclaredThrowableException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.ListIterator;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;
import com.exasol.swig.ResultHandler;
import com.exasol.swig.TableIterator;
import com.exasol.swig.ImportSpecificationWrapper;
import com.exasol.swig.ExportSpecificationWrapper;
import com.exasol.swig.ConnectionInformationWrapper;

class ExaWrapper {
    static byte[] runSingleCall(String fn, Object args) throws Throwable {
    Class argClass = Object.class;
        if (args != null) {
             if (args instanceof ImportSpecificationWrapper) {
                 ImportSpecificationWrapper is = (ImportSpecificationWrapper)args;
                 List<String> columnNames = new ArrayList<String>();
                 List<String> columnTypes = new ArrayList<String>();
                 if (is.isSubselect()) {
                     for (int i=0; i<is.numSubselectColumns(); i++) {                         
                         columnNames.add(is.copySubselectColumnName(i));
                         columnTypes.add(is.copySubselectColumnType(i));
                     }
                 }
                 Map<String,String> parameters = new HashMap<String,String>();
                 for (int i=0; i<is.getNumberOfParameters(); i++) {
                     parameters.put(is.copyKey(i),is.copyValue(i));
                 }
                 ConnectionInformationWrapper w = is.hasConnectionInformation()?is.getConnectionInformation():null;

                 args = new ExaImportSpecificationImpl(is.isSubselect(),
                                                       columnNames,
                                                       columnTypes,
                                                       is.hasConnectionName()?is.copyConnectionName():null,
                                                       (w!=null)?new ExaConnectionInformationImpl(w.copyKind(), w.copyAddress(), w.copyUser(), w.copyPassword()):null,
                                                       parameters);
                 argClass = ExaImportSpecification.class;
             } else if (args instanceof ExportSpecificationWrapper) {
                 ExportSpecificationWrapper es = (ExportSpecificationWrapper)args;
                 Map<String,String> parameters = new HashMap<String,String>();
                 for (int i=0; i<es.getNumberOfParameters(); i++) {
                     parameters.put(es.copyKey(i),es.copyValue(i));
                 }
                 ConnectionInformationWrapper w = es.hasConnectionInformation()?es.getConnectionInformation():null;
                 List<String> sourceColumnNames = new ArrayList<String>();
                 for (int i=0; i<es.numSourceColumns(); i++) {
                     sourceColumnNames.add(es.copySourceColumnName(i));
                 }

                 args = new ExaExportSpecificationImpl(es.hasConnectionName()?es.copyConnectionName():null,
                                                       (w!=null)?new ExaConnectionInformationImpl(w.copyKind(), w.copyAddress(), w.copyUser(), w.copyPassword()):null,
                                                       parameters,
                                                       es.hasTruncate(), es.hasReplace(),
                                                       es.copyCreatedBy(), sourceColumnNames);
                 argClass = ExaExportSpecification.class;
             } else if (fn.equals("adapterCall")) { // TODO VS This will be refactored completely soon
                 // args is already a String with the String arg, so nothing to do
                 argClass = String.class;
             } else {
                 throw new ExaCompilationException("F-UDF-CL-SL-JAVA-1065: Internal error: single call argument with unknown DTO: " + args.toString());
             }
        }

        ExaMetadataImpl exaMetadata = new ExaMetadataImpl();
        String exMsg = exaMetadata.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException("UDF.CL.SL.JAVA-1161: "+exMsg);
        }
        // Take the scriptClass name specified by user in the script, or the name of the script as fallback
        boolean userDefinedScriptName = true;
        String scriptClassName = System.getProperty("exasol.scriptclass", "");
        if (scriptClassName.trim().isEmpty()) {
            userDefinedScriptName = false;
            scriptClassName = exaMetadata.getScriptName();
            scriptClassName = scriptClassName.replace('.', '_');  // ** see comment in run() method
            scriptClassName = "com.exasol." + scriptClassName;
        }
        try {
            Class<?> scriptClass = Class.forName(scriptClassName);
            if (args == null) {
                Class[] params = {ExaMetadata.class};
                Method method = scriptClass.getDeclaredMethod(fn, params);
                String resS = String.valueOf(method.invoke(null, exaMetadata));
                return resS.getBytes("UTF-8");
            } else {
                Class[] params = {ExaMetadata.class,argClass};
                Method method = scriptClass.getDeclaredMethod(fn, params);
                String resS = String.valueOf(method.invoke(null, exaMetadata, args));
                return resS.getBytes("UTF-8");
            }
        } catch (java.lang.ClassNotFoundException ex) {
            if (userDefinedScriptName) {
                throw new ExaCompilationException("F-UDF-CL-SL-JAVA-1066: The main script class defined via %scriptclass cannot be found: " + scriptClassName);
            } else {
                throw new ExaCompilationException("F-UDF-CL-SL-JAVA-1067: The main script class (same name as the script) cannot be found: " + scriptClassName + ". Please create the class or specify the class via %scriptclass.");
            }
        } catch (InvocationTargetException ex) {
              throw convertReflectiveExceptionToCause("F-UDF-CL-SL-JAVA-1068","Exception during singleCall "+fn,ex);
        } catch (NoSuchMethodException ex) {
           throw new ExaUndefinedSingleCallException(fn);
        }
    }

    static Class<?> getScriptClass(ExaMetadataImpl exaMetadata) throws Throwable {
        boolean userDefinedScriptName = true;
        String scriptClassName = System.getProperty("exasol.scriptclass", "");
        if (scriptClassName.trim().isEmpty()) {
            userDefinedScriptName = false;
            scriptClassName = exaMetadata.getScriptName();
            // Only for test simulator (e.g., script.java -> script_java)
            scriptClassName = scriptClassName.replace('.', '_');
            scriptClassName = "com.exasol." + scriptClassName;
        }
        try{
            // Take the scriptClass name specified by user in the script, or the name of the script as fallback
            Class<?> scriptClass = Class.forName(scriptClassName);
            return scriptClass;
        } catch (java.lang.ClassNotFoundException ex) {
            if (userDefinedScriptName) {
                throw new ExaCompilationException("F-UDF.CL.SL.JAVA-1072: The main script class defined via %scriptclass cannot be found: " + scriptClassName);
            } else {
                throw new ExaCompilationException("F-UDF.CL.SL.JAVA-1073: The main script class (same name as the script) cannot be found: " + scriptClassName + ". Please create the class or specify the class via %scriptclass.");
            }
        }
    }

  
    static ExaMetadataImpl getMetaData() throws Throwable {
        ExaMetadataImpl exaMetadata = new ExaMetadataImpl();
        String exMsg = exaMetadata.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException("UDF.CL.SL.JAVA-1165: "+exMsg);
        }
        return exaMetadata;
    }

    static void run() throws Throwable {
        ExaMetadataImpl exaMetadata = null;
        try{
            exaMetadata = getMetaData();
        }catch(ExaIterationException ex){
            throw new ExaIterationException("F-UDF.CL.SL.JAVA-1069: "+ex.getMessage());
        }
        TableIterator tableIterator = new TableIterator();
        String exMsg = tableIterator.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException("F-UDF-CL-SL-JAVA-1070: "+exMsg);
        }
        ResultHandler resultHandler = new ResultHandler(tableIterator);
        exMsg = resultHandler.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException("F-UDF-CL-SL-JAVA-1071: "+exMsg);
        }

        ExaIteratorImpl exaIter = new ExaIteratorImpl(exaMetadata, tableIterator, resultHandler);

        Class<?> scriptClass = null;
        try {
            scriptClass = getScriptClass(exaMetadata);
        } catch (ExaCompilationException ex){
            throw new ExaCompilationException("F-UDF.CL.SL.JAVA-1165: "+ex.getMessage());
        }
        // init()
        try {
            Class[] initParams = {ExaMetadata.class};
            Method initMethod = scriptClass.getDeclaredMethod("init", initParams);
            initMethod.invoke(null, exaMetadata);
        } catch (InvocationTargetException ex) {
            throw convertReflectiveExceptionToCause("F-UDF-CL-SL-JAVA-1074","Exception during init",ex);
        } catch (NoSuchMethodException ex) { 
            System.err.println("W-UDF-CL-SL-JAVA-1075: Skipping init, because init method cannot be found.");
        }

        // run()
        Class[] runParams = {ExaMetadata.class, ExaIterator.class};
        Method runMethod = scriptClass.getDeclaredMethod("run", runParams);

        try {
            if (exaMetadata.getInputType().equals("SET")) { // MULTIPLE INPUT
                if (exaMetadata.getOutputType().equals("EMIT")) { // MULTIPLE OUTPUT
                    if (!runMethod.getReturnType().equals(Void.TYPE))
                        throw new ExaCompilationException("F-UDF-CL-SL-JAVA-1076: EMITS requires a void return type for run()");
                    exaIter.setInsideRun(true);
                    Object returnValue = runMethod.invoke(null, exaMetadata, exaIter);
                    exaIter.setInsideRun(false);
                }
                else { // EXACTLY_ONCE OUTPUT
                    if (runMethod.getReturnType().equals(Void.TYPE))
                        throw new ExaCompilationException("F-UDF-CL-SL-JAVA-1077: RETURNS requires a non-void return type for run()");
                    exaIter.setInsideRun(true);
                    Object returnValue = runMethod.invoke(null, exaMetadata, exaIter);
                    exaIter.setInsideRun(false);
                    exaIter.emit(returnValue);
                }
            }
            else { // EXACTLY_ONCE INPUT
                if (exaMetadata.getOutputType().equals("EMIT")) { // MULTIPLE OUTPUT
                    if (!runMethod.getReturnType().equals(Void.TYPE))
                        throw new ExaCompilationException("F-UDF-CL-SL-JAVA-1078: EMITS requires a void return type for run()");
                    do {
                        exaIter.setInsideRun(true);
                        Object returnValue = runMethod.invoke(null, exaMetadata, exaIter);
                        exaIter.setInsideRun(false);
                    } while (exaIter.next());
                }
                else { // EXACTLY_ONCE OUTPUT
                    if (runMethod.getReturnType().equals(Void.TYPE))
                        throw new ExaCompilationException("F-UDF-CL-SL-JAVA-1079: RETURNS requires a non-void return type for run()");
                    do {
                        exaIter.setInsideRun(true);
                        Object returnValue = runMethod.invoke(null, exaMetadata, exaIter);
                        exaIter.setInsideRun(false);
                        exaIter.emit(returnValue);
                    } while (exaIter.next());
                }
            }
        }
        catch (InvocationTargetException ex) {
            throw convertReflectiveExceptionToCause("F-UDF-CL-SL-JAVA-1080","Exception during run",ex);
        }

        resultHandler.flush();
    }

    static void cleanup() throws Throwable {
        // FIXME cleanup gets only called if run is successful
        // cleanup()
        ExaMetadataImpl exaMetadata = null;
        try{
            exaMetadata = getMetaData();
        }catch(ExaIterationException ex){
            throw new ExaIterationException("F-UDF.CL.SL.JAVA-1166: "+ex.getMessage());
        }
        Class<?> scriptClass = null;
        try {
            scriptClass = getScriptClass(exaMetadata);
        } catch (ExaCompilationException ex){
            throw new ExaCompilationException("F-UDF.CL.SL.JAVA-1167: "+ex.getMessage());
        }
        try {
            Class[] cleanupParams = {ExaMetadata.class};
            Method cleanupMethod = scriptClass.getDeclaredMethod("cleanup", cleanupParams);
            cleanupMethod.invoke(null, exaMetadata);
        }
        catch (InvocationTargetException ex) {
            throw convertReflectiveExceptionToCause("F-UDF-CL-SL-JAVA-1081","Exception during cleanup",ex);
        }
        catch (NoSuchMethodException ex) {
            System.err.println("W-UDF.CL.SL.JAVA-1082: Skipping init, because cleanup method cannot be found.");
        }
    }

    private static String cleanStackTrace(Throwable ex){
        Throwable exc = ex;
        while (exc != null && (exc instanceof InvocationTargetException ||
                    exc instanceof MalformedParameterizedTypeException ||
                    exc instanceof UndeclaredThrowableException)) {
            Throwable cause = exc.getCause();
            if (cause == null)
                break;
            else
                exc = cause;
        }

        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        exc.printStackTrace(pw);
        String stacktrace = sw.toString();
            LinkedList<String> stacktrace_lines = new LinkedList<String>(Arrays.asList(stacktrace.split("\\r?\\n")));

        ListIterator list_Iter = stacktrace_lines.listIterator(0);
        while (list_Iter.hasNext()) {
            String line = (String) list_Iter.next();
            list_Iter.set(line.replaceFirst("^\tat ", ""));
        }
        list_Iter = stacktrace_lines.listIterator(stacktrace_lines.size());
        list_Iter = stacktrace_lines.listIterator(stacktrace_lines.size());
        Integer start_index = null;

        while (list_Iter.hasPrevious()) {
            Integer index = list_Iter.previousIndex();
            String line = (String) list_Iter.previous();
            if (line.startsWith("com.exasol.Exa")) {
                if (start_index == null) {
                    start_index = index;
                }
            } else if (line.startsWith("java.base/")) {
                if (start_index != null &&
                        (line.startsWith("java.base/jdk.internal.reflect") ||
                                line.startsWith("java.base/java.lang.reflect"))) {
                    list_Iter.remove();
                }
            } else {
                start_index = null;
            }
        }

        list_Iter = stacktrace_lines.listIterator(0);
        StringWriter stringWriter = new StringWriter();
        PrintWriter writer = new PrintWriter(stringWriter, true);
        while (list_Iter.hasNext()) {
            String line = (String) list_Iter.next();
            writer.println(line);
        }
        String cleanedStacktrace = stringWriter.toString();
        return cleanedStacktrace;
    }

    private static Throwable convertReflectiveExceptionToCause(String error_code, String errorMessage, Throwable ex) {
        String cleanedStacktrace = cleanStackTrace(ex); 
        String error_message=error_code+": "+errorMessage+" \n"+cleanedStacktrace;
        System.out.println(error_message);
        return new ExaUDFException(error_message);
    }
}
