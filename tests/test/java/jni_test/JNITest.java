package com.exasol.udf.java.test;
/**
 * http://www.java-tips.org/other-api-tips/jni/simple-example-of-using-the-java-native-interface.html
 */

class JNITest {
  static {
    System.loadLibrary("JNITest");
  }

  private native String runTest();

  //application main entry point
  public static void main(String[] args) {

    //invoke non-static print method
    String test_output = new JNITest().runTest();
    System.out.println("test_output: "+test_output);
  }
}
