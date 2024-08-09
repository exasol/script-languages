#ifndef SWIG_RESULT_HANDLER_H
#define SWIG_RESULT_HANDLER_H

#include <cstdint>
#include <stdlib.h>
#include <sstream>
#include "base/exaudflib/load_dynamic.h"
#include "base/exaudflib/swig/swig_table_iterator.h"

#define SWIG_MAX_VAR_DATASIZE 4000000

namespace SWIGVMContainers {

class SWIGRAbstractResultHandler {
public:
    virtual ~SWIGRAbstractResultHandler() {};
    virtual void reinitialize()=0;
    virtual unsigned long rowsEmited()=0;
    virtual void flush()=0;
    virtual bool next()=0;
    virtual void setDouble(unsigned int col, const double v)=0;
    virtual void setString(unsigned int col, const char *v, size_t l)=0;
    virtual void setInt32(unsigned int col, const int32_t v)=0;
    virtual void setInt64(unsigned int col, const int64_t v)=0;
    virtual void setNumeric(unsigned int col, const char *v)=0;
    virtual void setTimestamp(unsigned int col, const char *v)=0;
    virtual void setDate(unsigned int col, const char *v)=0;
    virtual void setBoolean(unsigned int col, const bool v)=0;
    virtual void setNull(unsigned int col)=0;
    virtual const char* checkException()=0;
};


class SWIGResultHandler { //: public SWIGRAbstractResultHandler {
    SWIGRAbstractResultHandler* impl=nullptr;
    typedef SWIGVMContainers::SWIGRAbstractResultHandler* (*CREATE_RESULTHANDLER_FUN)(SWIGVMContainers::SWIGTableIterator*);
public:
    SWIGResultHandler(SWIGTableIterator* table_iterator)
    {
#ifndef UDF_PLUGIN_CLIENT
        CREATE_RESULTHANDLER_FUN creator = (CREATE_RESULTHANDLER_FUN)load_dynamic("create_SWIGResultHandler");
        impl = creator(table_iterator);
#else
        impl = create_SWIGResultHandler(table_iterator);
#endif
    }

    virtual ~SWIGResultHandler() {
        if (impl!=nullptr) {
            delete impl;
        }
    }
    virtual void reinitialize() {impl->reinitialize(); }
    virtual unsigned long rowsEmited() {return impl->rowsEmited();}
    virtual void flush() {impl->flush();}
    virtual bool next() {return impl->next();}
    virtual void setDouble(unsigned int col, const double v) {impl->setDouble(col, v);}
    virtual void setString(unsigned int col, const char *v, size_t l) {impl->setString(col,v,l);}
    virtual void setInt32(unsigned int col, const int32_t v) {impl->setInt32(col,v);}
    virtual void setInt64(unsigned int col, const int64_t v) {impl->setInt64(col,v);}
    virtual void setNumeric(unsigned int col, const char *v) {impl->setNumeric(col,v);}
    virtual void setTimestamp(unsigned int col, const char *v) {impl->setTimestamp(col,v);}
    virtual void setDate(unsigned int col, const char *v) {impl->setDate(col,v);}
    virtual void setBoolean(unsigned int col, const bool v) {impl->setBoolean(col,v);}
    virtual void setNull(unsigned int col) {impl->setNull(col);}
    const char* checkException() {return impl->checkException();}
};

} //namespace SWIGVMContainers

#endif //SWIG_RESULT_HANDLER_H