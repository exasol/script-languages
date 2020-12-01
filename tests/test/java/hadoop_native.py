#!/usr/bin/env python2.7

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import useData, expectedFailure

class JAVA_NATIVE(udf.TestCase):

    def setUp(self):
        self.query('CREATE SCHEMA JAVA_NATIVE', ignore_errors=True)
        self.query('OPEN SCHEMA JAVA_NATIVE')

    def test_java_native_net(self):
        self.query(udf.fixindent('''
                CREATE JAVA SET SCRIPT JAVA_NATIVE_NET (a varchar(100)) EMITS (b varchar(10000)) AS
                %env LD_LIBRARY_PATH=/tmp/;
                %jvmoption -verbose:jni;
                import java.net.NetworkInterface;
                import java.io.*;
                import java.lang.reflect.Field;
                import java.util.Map;
                import java.util.Arrays;

                class JAVA_NATIVE_NET {

                  // From https://github.com/exasol/hadoop-etl-udfs/blob/0ec2dddf811d1dc5eaad6e7fa16d2da410cfe4b8/hadoop-etl-common/src/main/java/com/exasol/hadoop/NativeHadoopLibUtils.java
                  static String addDirToJavaLibraryPath(String path) throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
                    System.setProperty("java.library.path", path );
                    Field fieldSysPath = ClassLoader.class.getDeclaredField( "sys_paths" );
                    fieldSysPath.setAccessible( true );
                    String[] sys_paths = (String[])fieldSysPath.get(null);
                    String result = "";
                    if(sys_paths == null){
                        result = "field sys_paths: null";
                    }else{
                        result = "field sys_paths: "+Arrays.toString(sys_paths);
                    }
                    String[] new_sys_paths = Arrays.copyOf(sys_paths, sys_paths.length+1);
                    new_sys_paths[sys_paths.length] = path;
                    fieldSysPath.set( null, new_sys_paths );
                    //fieldSysPath.set( null, null );
                    return result; 
                  }


                  static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    ctx.emit("start");
                    String result = addDirToJavaLibraryPath("/tmp");
                    ctx.emit(result);
                    try {
                        final String jvmName = NetworkInterface.getNetworkInterfaces().nextElement().getName();
                        ctx.emit(jvmName);
                        ctx.emit("SUCCESS");
                    } catch (Throwable e) { 
                        StringWriter sw = new StringWriter();
                        PrintWriter pw = new PrintWriter(sw);
                        e.printStackTrace(pw);
                        ctx.emit(sw.toString());
                        ctx.emit("FAILED");
                    }
                  }
                }
                '''))
        rows = self.query('''
            SELECT JAVA_NATIVE_NET('');
            ''')
        print(rows)
        self.fail()

if __name__ == '__main__':
    udf.main()

