# removed the -c option because we need to link this
all: header lib class tar
tar:
	tar -C output/ -cvzf jni_test.tar.gz .
lib:
	gcc -shared -Wall -Werror -I/usr/lib/jvm/java-11-openjdk-amd64/include/ -I/usr/lib/jvm/java-11-openjdk-amd64/include/linux -o libJNITest.so -fPIC com_exasol_udf_java_test_JNITest.c
	mkdir -p output || echo "Directory exists, do nothing"
	mv libJNITest.so output
header:
	javac -h . JNITest.java
class:
	javac JNITest.java
	mkdir -p output/com/exasol/udf/java/test || echo "Directory exists, do nothing"
	mv JNITest.class output/com/exasol/udf/java/test	
clean:
	rm -rf output 
run:
	java -Djava.library.path=output -cp output com.exasol.udf.java.test.JNITest
