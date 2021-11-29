import os
import sys

JAVA_SRC_PATH = 'java_src'


def filter_swig_code(target, source):
    assert len(target) == 1 and len(source) == 1

    builddir = './'
    # if 'build_dir' in env:
    #     builddir = os.path.join( env['build_dir'], SWIGBASED_DIR )

    # try:
    #     RE
    # except:
    #     RE = env['CURRENTTOOLCHAIN']

    target = str(target[0])
    source = str(source[0])
    print("filter_swig_code: reading file ", str(source))
    output = []
    with open(source) as rofile:
        for line in rofile:
            line = line.rstrip()
            if source.endswith('_python_tmp.cc'):
                line = line.replace('SWIG_init(void)', 'init_exascript_python(void)')
                if line == '  PyObject *m, *d, *md;':
                    line = '  PyObject *m, *d;'
                elif line == '  md = d = PyModule_GetDict(m);':
                    line = '  d = PyModule_GetDict(m);'
                elif line == '  (void)md;':
                    line = ''
                elif line == '  PyDict_SetItemString(md, "__all__", public_interface);':
                    line = '  PyDict_SetItemString(d, "__all__", public_interface);'
            elif source.endswith('_r_tmp.cc') or source.endswith('_r_tmp.h'):
                if line == '  const char *p = typeName;':
                    output.append('#if 0')
                elif line == '     p = typeName + 1;':
                    output.append(line)
                    line = '#endif'
                elif line == '  delete arg1;':
                    line = '  if (argp1 != NULL) delete arg1;'
                elif line == '   {NULL, NULL, 0}':
                    output.append('    {"RVM_next_block", (DL_FUNC) &RVM_next_block, 4},')
                    output.append('    {"RVM_emit_block", (DL_FUNC) &RVM_emit_block, 2},')
                elif line == 'SWIGINTERN R_CallMethodDef CallEntries[] = {':
                    output.append('extern "C" SEXP RVM_next_block(SEXP dataexp);')
                    output.append('extern "C" SEXP RVM_emit_block(SEXP dataexp);')
            elif source.endswith('_java_tmp.cc'):
                if line == 'SWIGEXPORT jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getString(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2) {':
                    output.append('SWIGEXPORT jbyteArray JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getStringByteArray(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2) {')
                    line = next(rofile).rstrip()
                    if line == '  jstring jresult = 0 ;':
                        output.append('  jbyteArray jresult = 0 ;')
                        line = next(rofile).rstrip()
                    while line != '  if (result) jresult = jenv->NewStringUTF((const char *)result);':
                        output.append(line)
                        line = next(rofile).rstrip()
                    if line == '  if (result) jresult = jenv->NewStringUTF((const char *)result);':
                        output.append('  if (result) {')
                        output.append('    jsize len = strlen(result);')
                        output.append('    jresult = jenv->NewByteArray(len);')
                        output.append('    jenv->SetByteArrayRegion(jresult, 0, len, (jbyte *)result);')
                        output.append('  }')
                        output.append('  return jresult;')
                        while line != '}':
                            line = next(rofile).rstrip()
                if line == 'SWIGEXPORT void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setString(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jstring jarg3, jlong jarg4) {':
                    output.append('SWIGEXPORT void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setStringByteArray(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jbyteArray jarg3, jlong jarg4) {')
                    line = next(rofile).rstrip()
                    while line != '  if (jarg3) {':
                        output.append(line)
                        line = next(rofile).rstrip()
                    if line == '  if (jarg3) {':
                        output.append('  arg4 = (size_t)jarg4;')
                        output.append('  if (jarg3) {')
                        output.append('    arg3 = new char[arg4 + 1];')
                        output.append('    if (!arg3) return;')
                        output.append('    jenv->GetByteArrayRegion(jarg3, 0, jarg4, (jbyte *)arg3);')
                        output.append('    arg3[arg4] = 0;')
                        output.append('  }')
                        output.append('  (arg1)->setString(arg2,arg3,arg4);')
                        output.append('  if (arg3) delete [] arg3;')
                        while line != '}':
                            line = next(rofile).rstrip()
            output.append(line)
    with open(str(target), 'w') as fd:
        fd.write('\n'.join(output))
        fd.write('\n')
    print("filter_swig_code: wrote file ", str(target))
    # piggyback on exascript_java.cc target to patch *.java files (avoids error from multiple ways to build target)
    if source.endswith('_java_tmp.cc'):
        filter_java_swig_code(builddir)
    return 0


def filter_java_swig_code(builddir=''):
    files = map(lambda s: os.path.join(builddir+f'{JAVA_SRC_PATH}/com/exasol/swig', s),
                ['TableIterator.java', 'ResultHandler.java', 'exascript_javaJNI.java'])
    for target in files:
        output = []
        with open(target) as target_file:
            for line in target_file:
                line = line.rstrip()
                if target.endswith('TableIterator.java'):
                    if line == '  public String getString(long col) {':
                        line = '  public byte[] getString(long col) {'
                    elif line == '    return exascript_javaJNI.TableIterator_getString(swigCPtr, this, col);':
                        line = '    return exascript_javaJNI.TableIterator_getStringByteArray(swigCPtr, this, col);'
                elif target.endswith('ResultHandler.java'):
                    if line == '  public void setString(long col, String v, long l) {':
                        line = '  public void setString(long col, byte[] v) {'
                    elif line == '    exascript_javaJNI.ResultHandler_setString(swigCPtr, this, col, v, l);':
                        line = '    exascript_javaJNI.ResultHandler_setStringByteArray(swigCPtr, this, col, v, v.length);'
                elif target.endswith('exascript_javaJNI.java'):
                    if line == '  public final static native String TableIterator_getString(long jarg1, TableIterator jarg1_, long jarg2);':
                        line = '  public final static native byte[] TableIterator_getStringByteArray(long jarg1, TableIterator jarg1_, long jarg2);'
                    elif line == '  public final static native void ResultHandler_setString(long jarg1, ResultHandler jarg1_, long jarg2, String jarg3, long jarg4);':
                        line = '  public final static native void ResultHandler_setStringByteArray(long jarg1, ResultHandler jarg1_, long jarg2, byte[] jarg3, long jarg4);'
                output.append(line)
        with open(str(target), 'w') as fd:
            fd.write('\n'.join(output))
            fd.write('\n')
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: filter_swig_code.py target source')
        sys.exit(1)
    print("filtering swig code ", str([sys.argv[1]]), str([sys.argv[2]]))
    filter_swig_code([sys.argv[1]], [sys.argv[2]])
