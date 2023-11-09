#ifndef SWIG_TABLE_ITERATOR_H
#define SWIG_TABLE_ITERATOR_H

#include <cstdint>
#include <stdlib.h>
#include <sstream>
#include "exaudflib/load_dynamic.h"

namespace SWIGVMContainers {

class AbstractSWIGTableIterator {
public:
    virtual ~AbstractSWIGTableIterator() {}
    virtual void reinitialize()=0;
    virtual bool next()=0;
    virtual bool eot()=0;
    virtual void reset()=0;
    virtual unsigned long restBufferSize()=0;
    virtual unsigned long rowsCompleted()=0;
    virtual unsigned long rowsInGroup()=0;
    virtual double getDouble(unsigned int col)=0;
    virtual const char *getString(unsigned int col, size_t *length = NULL)=0;
    virtual const char *getBinary(unsigned int col, size_t *length = NULL)=0;
    virtual int32_t getInt32(unsigned int col)=0;
    virtual int64_t getInt64(unsigned int col)=0;
    virtual const char *getNumeric(unsigned int col)=0;
    virtual const char *getTimestamp(unsigned int col)=0;
    virtual const char *getDate(unsigned int col)=0;
    virtual bool getBoolean(unsigned int col)=0;
    virtual bool wasNull()=0;
    virtual uint64_t get_current_row()=0;
    virtual const char* checkException()=0;
};



class SWIGTableIterator { //: public AbstractSWIGTableIterator {
    typedef SWIGVMContainers::AbstractSWIGTableIterator* (*CREATE_TABLEITERATOR_FUN)();

    AbstractSWIGTableIterator* impl=nullptr;
public:
    SWIGTableIterator()
    {
#ifndef UDF_PLUGIN_CLIENT
        CREATE_TABLEITERATOR_FUN creator = (CREATE_TABLEITERATOR_FUN)load_dynamic("create_SWIGTableIterator");
        impl = creator();
#else
        impl = create_SWIGTableIterator();
#endif
    }

    virtual ~SWIGTableIterator() {
        if (impl!=nullptr) {
            delete impl;
        }
    }
    virtual void reinitialize() { impl->reinitialize();}
    virtual bool next() { return impl->next(); }
    virtual bool eot() { return impl->eot(); }
    virtual void reset() { return impl->reset(); }
    virtual unsigned long restBufferSize() { return impl->restBufferSize();}
    virtual unsigned long rowsCompleted() { return impl->rowsCompleted();}
    virtual unsigned long rowsInGroup() { return impl->rowsInGroup();}
    virtual double getDouble(unsigned int col) { return impl->getDouble(col);}
    virtual const char *getString(unsigned int col, size_t *length = NULL) {return impl->getString(col, length);}
    virtual const char *getBinary(unsigned int col, size_t *length = NULL) {return impl->getBinary(col, length);}
    virtual int32_t getInt32(unsigned int col) {return impl->getInt32(col);}
    virtual int64_t getInt64(unsigned int col) {return impl->getInt64(col);}
    virtual const char *getNumeric(unsigned int col) {return impl->getNumeric(col);}
    virtual const char *getTimestamp(unsigned int col) {return impl->getTimestamp(col);}
    virtual const char *getDate(unsigned int col) {return impl->getDate(col);}
    virtual bool getBoolean(unsigned int col) {return impl->getBoolean(col);}
    virtual bool wasNull() { return impl->wasNull();}
    virtual uint64_t get_current_row() {return impl->get_current_row();}
    const char* checkException() {return impl->checkException();}
};

} //namespace SWIGVMContainers

#endif //SWIG_TABLE_ITERATOR_H
