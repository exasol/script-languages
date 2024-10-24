#ifndef SCRIPTOPTIONLINEPARSERCONVERTER_H
#define SCRIPTOPTIONLINEPARSERCONVERTER_H 1

#include <string>
#include <vector>
#include <functional>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Converter {

    public:
        Converter();
    
        void convertScriptClassName(const std::string & value);

        void convertJvmOption(const std::string & value);

        std::vector<std::string>&& moveJvmOptions() {
            return std::move(m_jvmOptions);
        }

        virtual void convertExternalJar(const std::string & value) = 0;

        virtual void iterateJarPaths(std::function<void(const std::string &option)> callback) const = 0;


    private:

        std::vector<std::string> m_jvmOptions;

        const std::string m_whitespace;
};



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H