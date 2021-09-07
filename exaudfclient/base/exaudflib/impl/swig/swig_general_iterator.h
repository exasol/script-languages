#ifndef EXAUDFCLIENT_SWIGGENERALITERATOR_H
#define EXAUDFCLIENT_SWIGGENERALITERATOR_H

#include "exaudflib/exaudflib.h"

namespace SWIGVMContainers {

class SWIGGeneralIterator {
protected:
    SWIGVMExceptionHandler *m_exch;
public:
    //        SWIGGeneralIterator(SWIGVMExceptionHandler *exch): m_exch(exch) { }
    SWIGGeneralIterator();
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
