%begin %{
#define SWIG_PYTHON_STRICT_BYTE_CHAR
%}

%{
#include "base/exaudflib/swig/swig_result_handler.h"
#include "base/exaudflib/swig/swig_meta_data.h"

using namespace SWIGVMContainers;
#include "base/exaudflib/swig/script_data_transfer_objects_wrapper.h"
%}

%ignore ExecutionGraph::ConnectionInformationWrapper::ConnectionInformationWrapper;
%include "exaudflib/script_data_transfer_objects_wrapper.h"
%newobject SWIGMetadata::connectionInformation(char*);
%newobject ConnectionInformationWrapper::copyKind();
%newobject ConnectionInformationWrapper::copyAddress();
%newobject ConnectionInformationWrapper::copyUser();
%newobject ConnectionInformationWrapper::copyPassword();

%newobject ImportSpecificationWrapper::copySubselectColumnName();
%newobject ImportSpecificationWrapper::copySubselectColumnType();
%newobject ImportSpecificationWrapper::copyConnectionName();
%newobject ImportSpecificationWrapper::copyKey();
%newobject ImportSpecificationWrapper::copyValue();

%newobject ExportSpecificationWrapper::copyConnectionName();
%newobject ExportSpecificationWrapper::copyKey();
%newobject ExportSpecificationWrapper::copyValue();
%newobject ExportSpecificationWrapper::copyCreatedBy();
%newobject ExportSpecificationWrapper::copySourceColumnName();

%rename(Metadata) SWIGMetadata;
%rename(TableIterator) SWIGTableIterator;
%rename(ResultHandler) SWIGResultHandler;

enum SWIGVM_datatype_e {
    UNSUPPORTED = 0,
    DOUBLE = 1,
    INT32 = 2,
    INT64 = 3,
    NUMERIC = 4,
    TIMESTAMP = 5,
    DATE = 6,
    STRING = 7,
    BOOLEAN = 8 };

enum SWIGVM_itertype_e {
    EXACTLY_ONCE = 1,
    MULTIPLE = 2
};

class SWIGMetadata {
    public:
        SWIGMetadata();

        inline const char* databaseName();
        inline const char* databaseVersion();
        inline const char* scriptName();
        inline const char* scriptSchema();
        inline const char* currentUser();
        inline const char* scopeUser();
        inline const char* currentSchema();
        inline const char* scriptCode();
        inline const char* moduleContent(const char* name);
        inline const ExecutionGraph::ConnectionInformationWrapper* connectionInformation(const char* connection_name);
        inline const unsigned long long sessionID();
        inline const const char *sessionID_S();
        inline const unsigned long statementID();
        inline const unsigned int nodeCount();
        inline const unsigned int nodeID();
        inline const unsigned long long vmID();
        inline const const char *vmID_S();
        inline const unsigned long long memoryLimit();

        inline unsigned int inputColumnCount();
        inline const char *inputColumnName(unsigned int col);
        inline const SWIGVM_datatype_e inputColumnType(unsigned int col);
        inline const char *inputColumnTypeName(unsigned int col);
        inline const unsigned int inputColumnSize(unsigned int col);
        inline const unsigned int inputColumnPrecision(unsigned int col);
        inline const unsigned int inputColumnScale(unsigned int col);
        inline const SWIGVM_itertype_e inputType();

        inline unsigned int outputColumnCount();
        inline const char *outputColumnName(unsigned int col);
        inline const SWIGVM_datatype_e outputColumnType(unsigned int col);
        inline const char *outputColumnTypeName(unsigned int col);
        inline const unsigned int outputColumnSize(unsigned int col);
        inline const unsigned int outputColumnPrecision(unsigned int col);
        inline const unsigned int outputColumnScale(unsigned int col);
        inline const SWIGVM_itertype_e outputType();

        inline const char* checkException();
        inline const char*  pluginLanguageName();
};

class SWIGTableIterator {
    public:
        SWIGTableIterator();
        inline const char* checkException();
        inline void reinitialize();
        inline bool next();
        inline bool eot();
        inline void reset();
        inline unsigned long restBufferSize();
        inline unsigned long rowsInGroup();
        inline unsigned long rowsCompleted();
        inline double getDouble(unsigned int col);
        inline const char *getString(unsigned int col);
        inline int getInt32(unsigned int col);
        inline long long getInt64(unsigned int col);
        inline const char *getNumeric(unsigned int col);
        inline const char *getDate(unsigned int col);
        inline const char *getTimestamp(unsigned int col);
        inline bool getBoolean(unsigned int col);
        inline bool wasNull();
};

class SWIGResultHandler {
    public:
        SWIGResultHandler(SWIGTableIterator* it);
        inline const char* checkException();
        inline void reinitialize();
        inline bool next();
        inline void flush();
        inline void setDouble(unsigned int col, double v);
        inline void setString(unsigned int col, char *v, size_t l);
        inline void setInt32(unsigned int col, int v);
        inline void setInt64(unsigned int col, long long v);
        inline void setNumeric(unsigned int col, char *v);
        inline void setDate(unsigned int col, char *v);
        inline void setTimestamp(unsigned int col, char *v);
        inline void setBoolean(unsigned int col, bool v);
        inline void setNull(unsigned int col);
};
