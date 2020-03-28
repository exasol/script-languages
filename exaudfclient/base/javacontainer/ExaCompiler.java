package com.exasol;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.io.StringWriter;
import javax.tools.SimpleJavaFileObject;
import javax.tools.JavaFileObject;
import javax.tools.ToolProvider;
import javax.tools.JavaCompiler;
import javax.tools.JavaCompiler.CompilationTask;
import java.net.URI;

class ExaCompiler {
    static void compile(String classname, String code, String classpath) throws ExaCompilationException {
        temporaryClasspath = classpath;
        JavaFileObject file = new JavaSource(classname, code);
        Iterable<? extends JavaFileObject> compilationUnits = Arrays.asList(file);
        List<String> optionList = new ArrayList<String>(Arrays.asList("-d", temporaryClasspath));

        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        StringWriter compilationOutput = new StringWriter();
        CompilationTask task = compiler.getTask(compilationOutput, null, null, optionList, null, compilationUnits);

        boolean success = false;
        try {	
            success = task.call();
        } catch (Exception e) {
            // ignore
        }
        if (!success) {
            throw new ExaCompilationException("F-UDF.CL.J-113: "+compilationOutput.toString());
        }
    }

    static void compile(String classname, String code) throws ExaCompilationException {
        compile(classname, code, temporaryClasspath);
    }

    private static class JavaSource extends SimpleJavaFileObject {
        final String code;

        JavaSource(String classname, String code) {
            super(URI.create("string:///" + classname.replace(".","/") + Kind.SOURCE.extension), Kind.SOURCE);
            this.code = code;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            return this.code;
        }
    }

    private static String temporaryClasspath;
}
