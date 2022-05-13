#ifndef EXAUDFCLIENT_SWIGGENERALITERATOR_H
#define EXAUDFCLIENT_SWIGGENERALITERATOR_H

#include "exaudflib/swig/swig_common.h"
#include "exaudflib/impl/global.h"
#include "exaudflib/impl/swig/swig_parameter.h"

namespace SWIGVMContainers {

class SWIGGeneralIterator {
protected:
    SWIGVMExceptionHandler *m_exch;
public:
    //        SWIGGeneralIterator(SWIGVMExceptionHandler *exch): m_exch(exch) { }
    SWIGGeneralIterator()
    : m_exch(exaudflib::global.SWIGVM_params_ref->exch)
    {}
    virtual ~SWIGGeneralIterator() { }
    inline const char* checkException() {
        if (m_exch->exthrowed) {
            m_exch->exthrowed = false;
            return m_exch->exmsg.c_str();
        } else return NULL;
    }
};

} //namespace SWIGVMContainers

#endif //EXAUDFCLIENT_SWIGGENERALITERATOR_H
