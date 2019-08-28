package com.exasol;

import java.io.IOError;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.MalformedParameterizedTypeException;
import java.lang.reflect.Method;
import java.lang.reflect.UndeclaredThrowableException;
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
                 throw new ExaCompilationException("Internal error: single call argument with unknown DTO: " + args.toString());
             }
        }

        ExaMetadataImpl exaMetadata = new ExaMetadataImpl();
        String exMsg = exaMetadata.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException(exMsg);
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
                throw new ExaCompilationException("The main script class defined via %scriptclass cannot be found: " + scriptClassName);
            } else {
                throw new ExaCompilationException("The main script class (same name as the script) cannot be found: " + scriptClassName + ". Please create the class or specify the class via %scriptclass.");
            }
        } catch (InvocationTargetException ex) {
              throw convertReflectiveExceptionToCause(ex);
        } catch (NoSuchMethodException ex) {
           throw new ExaUndefinedSingleCallException(fn);
        }
    }

    static void run() throws Throwable {
        ExaMetadataImpl exaMetadata = new ExaMetadataImpl();
        String exMsg = exaMetadata.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException(exMsg);
        }
        TableIterator tableIterator = new TableIterator();
        exMsg = tableIterator.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException(exMsg);
        }
        ResultHandler resultHandler = new ResultHandler(tableIterator);
        exMsg = resultHandler.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException(exMsg);
        }

        ExaIteratorImpl exaIter = new ExaIteratorImpl(exaMetadata, tableIterator, resultHandler);

        // Take the scriptClass name specified by user in the script, or the name of the script as fallback
        boolean userDefinedScriptName = true;
        String scriptClassName = System.getProperty("exasol.scriptclass", "");
        if (scriptClassName.trim().isEmpty()) {
            userDefinedScriptName = false;
            scriptClassName = exaMetadata.getScriptName();
            // Only for test simulator (e.g., script.java -> script_java)
            scriptClassName = scriptClassName.replace('.', '_');
            scriptClassName = "com.exasol." + scriptClassName;
        }
        // init()
        Class<?> scriptClass = null;
        try {
            scriptClass = Class.forName(scriptClassName);
            Class[] initParams = {ExaMetadata.class};
            Method initMethod = scriptClass.getDeclaredMethod("init", initParams);
            initMethod.invoke(null, exaMetadata);
        }
        catch (java.lang.ClassNotFoundException ex) {
            if (userDefinedScriptName) {
                throw new ExaCompilationException("The main script class defined via %scriptclass cannot be found: " + scriptClassName);
            } else {
                throw new ExaCompilationException("The main script class (same name as the script) cannot be found: " + scriptClassName + ". Please create the class or specify the class via %scriptclass.");
            }
        } catch (InvocationTargetException ex) {
            throw convertReflectiveExceptionToCause(ex);
        } catch (NoSuchMethodException ex) { /* not defined */ }

        // run()
        Class[] runParams = {ExaMetadata.class, ExaIterator.class};
        Method runMethod = scriptClass.getDeclaredMethod("run", runParams);

        try {
            if (exaMetadata.getInputType().equals("SET")) { // MULTIPLE INPUT
                if (exaMetadata.getOutputType().equals("EMIT")) { // MULTIPLE OUTPUT
                    if (!runMethod.getReturnType().equals(Void.TYPE))
                        throw new ExaCompilationException("EMITS requires a void return type for run()");
                    exaIter.setInsideRun(true);
                    Object returnValue = runMethod.invoke(null, exaMetadata, exaIter);
                    exaIter.setInsideRun(false);
                }
                else { // EXACTLY_ONCE OUTPUT
                    if (runMethod.getReturnType().equals(Void.TYPE))
                        throw new ExaCompilationException("RETURNS requires a non-void return type for run()");
                    exaIter.setInsideRun(true);
                    Object returnValue = runMethod.invoke(null, exaMetadata, exaIter);
                    exaIter.setInsideRun(false);
                    exaIter.emit(returnValue);
                }
            }
            else { // EXACTLY_ONCE INPUT
                if (exaMetadata.getOutputType().equals("EMIT")) { // MULTIPLE OUTPUT
                    do {
                        if (!runMethod.getReturnType().equals(Void.TYPE))
                            throw new ExaCompilationException("EMITS requires a void return type for run()");
                        exaIter.setInsideRun(true);
                        Object returnValue = runMethod.invoke(null, exaMetadata, exaIter);
                        exaIter.setInsideRun(false);
                    } while (exaIter.next());
                }
                else { // EXACTLY_ONCE OUTPUT
                    do {
                        if (runMethod.getReturnType().equals(Void.TYPE))
                            throw new ExaCompilationException("RETURNS requires a non-void return type for run()");
                        exaIter.setInsideRun(true);
                        Object returnValue = runMethod.invoke(null, exaMetadata, exaIter);
                        exaIter.setInsideRun(false);
                        exaIter.emit(returnValue);
                    } while (exaIter.next());
                }
            }
        }
        catch (InvocationTargetException ex) {
            throw convertReflectiveExceptionToCause(ex);
        }

        resultHandler.flush();

        // cleanup()
        try {
            Class[] cleanupParams = {ExaMetadata.class};
            Method cleanupMethod = scriptClass.getDeclaredMethod("cleanup", cleanupParams);
            cleanupMethod.invoke(null, exaMetadata);
        }
        catch (InvocationTargetException ex) {
            throw convertReflectiveExceptionToCause(ex);
        }
        catch (NoSuchMethodException ex) { /* not defined */ }
    }

    private static Throwable convertReflectiveExceptionToCause(Throwable ex) {
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
        return exc;
    }
}
