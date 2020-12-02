#include <jni.h>
#include <stdio.h>
#include "com_exasol_udf_java_test_JNITest.h"

JNIEXPORT jstring JNICALL Java_com_exasol_udf_java_test_JNITest_runTest
  (JNIEnv * env, jobject object){
	const char* msg = "SUCCESS";
	jstring result = (*env)->NewStringUTF(env,msg); 
	return result;
}
