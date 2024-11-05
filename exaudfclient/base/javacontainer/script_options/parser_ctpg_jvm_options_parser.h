#ifndef SCRIPTOPTIONLINEPARSERCTPGJVMOPTIONSPARSER_H
#define SCRIPTOPTIONLINEPARSERCTPGJVMOPTIONSPARSER_H 1

#include <vector>
#include <string>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

namespace JvmOptionsCTPG {

typedef std::vector<std::string> tJvmOptions;

void parseJvmOptions(const std::string & jvmOptions, tJvmOptions& result);



} //namespace CTPG

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCTPGSCRIPTIMPORTER_H
