#ifndef EXASCRIPT_JAVA_JNI_DECL_H
#define EXASCRIPT_JAVA_JNI_DECL_H

#ifdef __cplusplus
extern "C" {
#endif


// the signature definitions can be printed with 'javah -classpath . com.exasol.swig.exascript_javaJNI'
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ConnectionInformationWrapper_1copyKind(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ConnectionInformationWrapper_1copyAddress(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ConnectionInformationWrapper_1copyUser(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ConnectionInformationWrapper_1copyPassword(JNIEnv *, jclass, jlong, jobject);

jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_new_1ImportSpecificationWrapper(JNIEnv *jenv, jclass, jlong);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1isSubselect(JNIEnv *, jclass, jlong, jobject);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1numSubselectColumns(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copySubselectColumnName(JNIEnv *, jclass, jlong, jobject, jlong);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copySubselectColumnType(JNIEnv *, jclass, jlong, jobject, jlong);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1hasConnectionName(JNIEnv *, jclass, jlong, jobject);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1hasConnectionInformation(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copyConnectionName(JNIEnv *, jclass, jlong, jobject);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1getConnectionInformation(JNIEnv *, jclass, jlong, jobject);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1getNumberOfParameters(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copyKey(JNIEnv *, jclass, jlong, jobject, jlong);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copyValue(JNIEnv *, jclass, jlong, jobject, jlong);

jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_new_1ExportSpecificationWrapper(JNIEnv *, jclass, jlong);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasConnectionName(JNIEnv *, jclass, jlong, jobject);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasConnectionInformation(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copyConnectionName(JNIEnv *, jclass, jlong, jobject);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1getConnectionInformation(JNIEnv *, jclass, jlong, jobject);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1getNumberOfParameters(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copyKey(JNIEnv *, jclass, jlong, jobject, jlong);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copyValue(JNIEnv *, jclass, jlong, jobject, jlong);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasTruncate(JNIEnv *, jclass, jlong, jobject);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasReplace(JNIEnv *, jclass, jlong, jobject);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasCreatedBy(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copyCreatedBy(JNIEnv *, jclass, jlong, jobject);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1numSourceColumns(JNIEnv *, jclass, jlong, jobject);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copySourceColumnName(JNIEnv *, jclass, jlong, jobject, jlong);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_delete_1ExportSpecificationWrapper(JNIEnv *, jclass, jlong);

jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_UNSUPPORTED_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_DOUBLE_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_INT32_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_INT64_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_NUMERIC_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_TIMESTAMP_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_DATE_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_STRING_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_BOOLEAN_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_EXACTLY_1ONCE_1get(JNIEnv *jenv, jclass jcls);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_MULTIPLE_1get(JNIEnv *jenv, jclass jcls);

jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_new_1Metadata(JNIEnv *jenv, jclass jcls);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1databaseName(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1databaseVersion(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1scriptName(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1scriptSchema(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1currentUser(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1scopeUser(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1currentSchema(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1scriptCode(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1moduleContent(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jstring jarg2);
jobject JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1connectionInformation(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jstring jarg2);
jobject JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1sessionID(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1sessionID_1S(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1statementID(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1nodeCount(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1nodeID(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jobject JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1vmID(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1vmID_1S(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jobject JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1memoryLimit(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnCount(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnName(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnType(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnTypeName(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnSize(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnPrecision(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnScale(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputType(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnCount(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnName(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnType(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnTypeName(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnSize(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnPrecision(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnScale(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputType(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_Metadata_1checkException(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_delete_1Metadata(JNIEnv *jenv, jclass jcls, jlong jarg1);

jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_new_1TableIterator(JNIEnv *jenv, jclass jcls);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1checkException(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1reinitialize(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1next(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1eot(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1reset(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1restBufferSize(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1rowsInGroup(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1rowsCompleted(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jdouble JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getDouble(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jbyteArray JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getStringByteArray(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jint JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getInt32(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getInt64(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getNumeric(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getDate(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getTimestamp(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getBoolean(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1wasNull(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_delete_1TableIterator(JNIEnv *jenv, jclass jcls, jlong jarg1);

jlong JNICALL Java_com_exasol_swig_exascript_1javaJNI_new_1ResultHandler(JNIEnv *jenv, jclass jcls);
jstring JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1checkException(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1reinitialize(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
jboolean JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1next(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1flush(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setDouble(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jdouble jarg3);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setStringByteArray(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jbyteArray jarg3, jlong jarg4);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setInt32(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jint jarg3);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setInt64(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jlong jarg3);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setNumeric(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jstring jarg3);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setDate(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jstring jarg3);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setTimestamp(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jstring jarg3);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setBoolean(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2, jboolean jarg3);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setNull(JNIEnv *jenv, jclass jcls, jlong jarg1, jobject jarg1_, jlong jarg2);
void JNICALL Java_com_exasol_swig_exascript_1javaJNI_delete_1ResultHandler(JNIEnv *jenv, jclass jcls, jlong jarg1);

// This array is a mapping from the native java methods in exascript_javaJNI class to 
// The signature syntax can be printed via "javap -s -p exascript_javaJNI" in the folder containing the exascript_javaJNI.class
JNINativeMethod methods[] = {
    {(char *)"UNSUPPORTED_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_UNSUPPORTED_1get},
    {(char *)"DOUBLE_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_DOUBLE_1get},
    {(char *)"INT32_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_INT32_1get},
    {(char *)"INT64_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_INT64_1get},
    {(char *)"NUMERIC_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_NUMERIC_1get},
    {(char *)"TIMESTAMP_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TIMESTAMP_1get},
    {(char *)"DATE_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_DATE_1get},
    {(char *)"STRING_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_STRING_1get},
    {(char *)"BOOLEAN_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_BOOLEAN_1get},
    {(char *)"EXACTLY_ONCE_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_EXACTLY_1ONCE_1get},
    {(char *)"MULTIPLE_get", (char *)"()I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_MULTIPLE_1get},
    {(char *)"new_Metadata", (char *)"()J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_new_1Metadata},
    {(char *)"Metadata_databaseName", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1databaseName},
    {(char *)"Metadata_databaseVersion", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1databaseVersion},
    {(char *)"Metadata_scriptName", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1scriptName},
    {(char *)"Metadata_scriptSchema", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1scriptSchema},
    {(char *)"Metadata_currentUser", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1currentUser},
    {(char *)"Metadata_scopeUser", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1scopeUser},
    {(char *)"Metadata_currentSchema", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1currentSchema},
    {(char *)"Metadata_scriptCode", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1scriptCode},
    {(char *)"Metadata_moduleContent", (char *)"(JLcom/exasol/swig/Metadata;Ljava/lang/String;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1moduleContent},
    {(char *)"Metadata_connectionInformation", (char *)"(JLcom/exasol/swig/Metadata;Ljava/lang/String;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1connectionInformation},
    {(char *)"Metadata_sessionID", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/math/BigInteger;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1sessionID},
    {(char *)"Metadata_sessionID_S", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1sessionID_1S},
    {(char *)"Metadata_statementID", (char *)"(JLcom/exasol/swig/Metadata;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1statementID},
    {(char *)"Metadata_nodeCount", (char *)"(JLcom/exasol/swig/Metadata;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1nodeCount},
    {(char *)"Metadata_nodeID", (char *)"(JLcom/exasol/swig/Metadata;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1nodeID},
    {(char *)"Metadata_vmID", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/math/BigInteger;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1vmID},
    {(char *)"Metadata_vmID_S", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1vmID_1S},
    {(char *)"Metadata_memoryLimit", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/math/BigInteger;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1memoryLimit},
    {(char *)"Metadata_inputColumnCount", (char *)"(JLcom/exasol/swig/Metadata;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnCount},
    {(char *)"Metadata_inputColumnName", (char *)"(JLcom/exasol/swig/Metadata;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnName},
    {(char *)"Metadata_inputColumnType", (char *)"(JLcom/exasol/swig/Metadata;J)I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnType},
    {(char *)"Metadata_inputColumnTypeName", (char *)"(JLcom/exasol/swig/Metadata;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnTypeName},
    {(char *)"Metadata_inputColumnSize", (char *)"(JLcom/exasol/swig/Metadata;J)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnSize},
    {(char *)"Metadata_inputColumnPrecision", (char *)"(JLcom/exasol/swig/Metadata;J)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnPrecision},
    {(char *)"Metadata_inputColumnScale", (char *)"(JLcom/exasol/swig/Metadata;J)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputColumnScale},
    {(char *)"Metadata_inputType", (char *)"(JLcom/exasol/swig/Metadata;)I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1inputType},
    {(char *)"Metadata_outputColumnCount", (char *)"(JLcom/exasol/swig/Metadata;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnCount},
    {(char *)"Metadata_outputColumnName", (char *)"(JLcom/exasol/swig/Metadata;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnName},
    {(char *)"Metadata_outputColumnType", (char *)"(JLcom/exasol/swig/Metadata;J)I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnType},
    {(char *)"Metadata_outputColumnTypeName", (char *)"(JLcom/exasol/swig/Metadata;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnTypeName},
    {(char *)"Metadata_outputColumnSize", (char *)"(JLcom/exasol/swig/Metadata;J)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnSize},
    {(char *)"Metadata_outputColumnPrecision", (char *)"(JLcom/exasol/swig/Metadata;J)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnPrecision},
    {(char *)"Metadata_outputColumnScale", (char *)"(JLcom/exasol/swig/Metadata;J)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputColumnScale},
    {(char *)"Metadata_outputType", (char *)"(JLcom/exasol/swig/Metadata;)I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1outputType},
    {(char *)"Metadata_checkException", (char *)"(JLcom/exasol/swig/Metadata;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_Metadata_1checkException},
    {(char *)"delete_Metadata", (char *)"(J)V", (void *)&JNICALL Java_com_exasol_swig_exascript_1javaJNI_delete_1Metadata},

    {(char *)"new_TableIterator", (char *)"()J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_new_1TableIterator},
    {(char *)"TableIterator_checkException", (char *)"(JLcom/exasol/swig/TableIterator;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1checkException},
    {(char *)"TableIterator_reinitialize", (char *)"(JLcom/exasol/swig/TableIterator;)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1reinitialize},
    {(char *)"TableIterator_next", (char *)"(JLcom/exasol/swig/TableIterator;)Z", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1next},
    {(char *)"TableIterator_eot", (char *)"(JLcom/exasol/swig/TableIterator;)Z", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1eot},
    {(char *)"TableIterator_reset", (char *)"(JLcom/exasol/swig/TableIterator;)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1reset},
    {(char *)"TableIterator_restBufferSize", (char *)"(JLcom/exasol/swig/TableIterator;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1restBufferSize},
    {(char *)"TableIterator_rowsInGroup", (char *)"(JLcom/exasol/swig/TableIterator;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1rowsInGroup},
    {(char *)"TableIterator_rowsCompleted", (char *)"(JLcom/exasol/swig/TableIterator;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1rowsCompleted},
    {(char *)"TableIterator_getDouble", (char *)"(JLcom/exasol/swig/TableIterator;J)D", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getDouble},
    {(char *)"TableIterator_getStringByteArray", (char *)"(JLcom/exasol/swig/TableIterator;J)[B", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getStringByteArray},
    {(char *)"TableIterator_getInt32", (char *)"(JLcom/exasol/swig/TableIterator;J)I", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getInt32},
    {(char *)"TableIterator_getInt64", (char *)"(JLcom/exasol/swig/TableIterator;J)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getInt64},
    {(char *)"TableIterator_getNumeric", (char *)"(JLcom/exasol/swig/TableIterator;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getNumeric},
    {(char *)"TableIterator_getDate", (char *)"(JLcom/exasol/swig/TableIterator;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getDate},
    {(char *)"TableIterator_getTimestamp", (char *)"(JLcom/exasol/swig/TableIterator;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getTimestamp},
    {(char *)"TableIterator_getBoolean", (char *)"(JLcom/exasol/swig/TableIterator;J)Z", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1getBoolean},
    {(char *)"TableIterator_wasNull", (char *)"(JLcom/exasol/swig/TableIterator;)Z", (void *)&Java_com_exasol_swig_exascript_1javaJNI_TableIterator_1wasNull},
    {(char *)"delete_TableIterator", (char *)"(J)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_delete_1TableIterator},

    {(char *)"new_ResultHandler", (char *)"(JLcom/exasol/swig/TableIterator;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_new_1ResultHandler},
    {(char *)"ResultHandler_checkException", (char *)"(JLcom/exasol/swig/ResultHandler;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1checkException},
    {(char *)"ResultHandler_reinitialize", (char *)"(JLcom/exasol/swig/ResultHandler;)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1reinitialize},
    {(char *)"ResultHandler_next", (char *)"(JLcom/exasol/swig/ResultHandler;)Z", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1next},
    {(char *)"ResultHandler_flush", (char *)"(JLcom/exasol/swig/ResultHandler;)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1flush},
    {(char *)"ResultHandler_setDouble", (char *)"(JLcom/exasol/swig/ResultHandler;JD)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setDouble},
    {(char *)"ResultHandler_setStringByteArray", (char *)"(JLcom/exasol/swig/ResultHandler;J[BJ)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setStringByteArray},
    {(char *)"ResultHandler_setInt32", (char *)"(JLcom/exasol/swig/ResultHandler;JI)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setInt32},
    {(char *)"ResultHandler_setInt64", (char *)"(JLcom/exasol/swig/ResultHandler;JJ)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setInt64},
    {(char *)"ResultHandler_setNumeric", (char *)"(JLcom/exasol/swig/ResultHandler;JLjava/lang/String;)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setNumeric},
    {(char *)"ResultHandler_setDate", (char *)"(JLcom/exasol/swig/ResultHandler;JLjava/lang/String;)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setDate},
    {(char *)"ResultHandler_setTimestamp", (char *)"(JLcom/exasol/swig/ResultHandler;JLjava/lang/String;)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setTimestamp},
    {(char *)"ResultHandler_setBoolean", (char *)"(JLcom/exasol/swig/ResultHandler;JZ)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setBoolean},
    {(char *)"ResultHandler_setNull", (char *)"(JLcom/exasol/swig/ResultHandler;J)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ResultHandler_1setNull},
    {(char *)"delete_ResultHandler", (char *)"(J)V", (void *)&Java_com_exasol_swig_exascript_1javaJNI_delete_1ResultHandler},


    {(char *)"ConnectionInformationWrapper_copyKind",(char *)"(JLcom/exasol/swig/ConnectionInformationWrapper;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ConnectionInformationWrapper_1copyKind},
    {(char *)"ConnectionInformationWrapper_copyAddress",(char *)"(JLcom/exasol/swig/ConnectionInformationWrapper;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ConnectionInformationWrapper_1copyAddress},
    {(char *)"ConnectionInformationWrapper_copyUser",(char *)"(JLcom/exasol/swig/ConnectionInformationWrapper;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ConnectionInformationWrapper_1copyUser},
    {(char *)"ConnectionInformationWrapper_copyPassword",(char *)"(JLcom/exasol/swig/ConnectionInformationWrapper;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ConnectionInformationWrapper_1copyPassword},


    {(char *)"new_ImportSpecificationWrapper",(char*)"(J)J",(void*)&Java_com_exasol_swig_exascript_1javaJNI_new_1ImportSpecificationWrapper},
    {(char *)"ImportSpecificationWrapper_isSubselect",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;)Z", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1isSubselect},
    {(char *)"ImportSpecificationWrapper_numSubselectColumns",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1numSubselectColumns},
    {(char *)"ImportSpecificationWrapper_copySubselectColumnName",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copySubselectColumnName},
    {(char *)"ImportSpecificationWrapper_copySubselectColumnType",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copySubselectColumnType},
    {(char *)"ImportSpecificationWrapper_hasConnectionName",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;)Z", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1hasConnectionName},
    {(char *)"ImportSpecificationWrapper_hasConnectionInformation",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;)Z", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1hasConnectionInformation},
    {(char *)"ImportSpecificationWrapper_copyConnectionName",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copyConnectionName},
    {(char *)"ImportSpecificationWrapper_getConnectionInformation",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1getConnectionInformation},
    {(char *)"ImportSpecificationWrapper_getNumberOfParameters",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;)J", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1getNumberOfParameters},
    {(char *)"ImportSpecificationWrapper_copyKey",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copyKey},
    {(char *)"ImportSpecificationWrapper_copyValue",(char *)"(JLcom/exasol/swig/ImportSpecificationWrapper;J)Ljava/lang/String;", (void *)&Java_com_exasol_swig_exascript_1javaJNI_ImportSpecificationWrapper_1copyValue},


    {(char *)"new_ExportSpecificationWrapper",(char *)"(J)J",(void *)&Java_com_exasol_swig_exascript_1javaJNI_new_1ExportSpecificationWrapper},
    {(char *)"ExportSpecificationWrapper_hasConnectionName",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)Z",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasConnectionName},
    {(char *)"ExportSpecificationWrapper_hasConnectionInformation",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)Z",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasConnectionInformation},
    {(char *)"ExportSpecificationWrapper_copyConnectionName",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)Ljava/lang/String;",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copyConnectionName},
    {(char *)"ExportSpecificationWrapper_getConnectionInformation",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)J",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1getConnectionInformation},
    {(char *)"ExportSpecificationWrapper_getNumberOfParameters",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)J",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1getNumberOfParameters},
    {(char *)"ExportSpecificationWrapper_copyKey",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;J)Ljava/lang/String;",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copyKey},
    {(char *)"ExportSpecificationWrapper_copyValue",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;J)Ljava/lang/String;",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copyValue},
    {(char *)"ExportSpecificationWrapper_hasTruncate",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)Z",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasTruncate},
    {(char *)"ExportSpecificationWrapper_hasReplace",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)Z",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasReplace},
    {(char *)"ExportSpecificationWrapper_hasCreatedBy",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)Z",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1hasCreatedBy},
    {(char *)"ExportSpecificationWrapper_copyCreatedBy",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)Ljava/lang/String;",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copyCreatedBy},
    {(char *)"ExportSpecificationWrapper_numSourceColumns",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;)J",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1numSourceColumns},
    {(char *)"ExportSpecificationWrapper_copySourceColumnName",(char *)"(JLcom/exasol/swig/ExportSpecificationWrapper;J)Ljava/lang/String;",(void *)&Java_com_exasol_swig_exascript_1javaJNI_ExportSpecificationWrapper_1copySourceColumnName},
    {(char *)"delete_ExportSpecificationWrapper",(char *)"(J)V",(void *)&Java_com_exasol_swig_exascript_1javaJNI_delete_1ExportSpecificationWrapper},

};

#ifdef __cplusplus
}
#endif

#endif
