#ifndef SCRIPTOPTIONLINESEXCEPTION_H
#define SCRIPTOPTIONLINESEXCEPTION_H

#include <stdexcept>


namespace ExecutionGraph
{

class OptionParserException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};


} // namespace ExecutionGraph

#endif // SCRIPTOPTIONLINESEXCEPTION_H