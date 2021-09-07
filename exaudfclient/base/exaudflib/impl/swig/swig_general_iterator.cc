#include "exaudflib/impl/swig/swig_general_iterator.h"
#include "exaudflib/impl/exaudflib_global.h"

SWIGVMContainers::SWIGGeneralIterator::SWIGGeneralIterator()
    : m_exch(exaudflib::global.SWIGVM_params_ref->exch)
    {}