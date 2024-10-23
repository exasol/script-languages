#ifndef SCRIPTOPTIONLINEPARSERCONVERTER_H
#define SCRIPTOPTIONLINEPARSERCONVERTER_H 1

#include <string>
#include <vector>
#include <functional>
#include <iterator>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Converter {

    public:
        Converter();
    
        virtual void convertExternalJar(const std::string & value) = 0;

        void convertScriptClassName(const std::string & value);

        void convertJvmOption(const std::string & value);

        virtual void iterateJarPaths(std::function<void(const std::string &option)> callback) const = 0;

        std::vector<std::string>&& moveJvmOptions() {
            return std::move(m_jvmOptions);
        }

    protected:
        const std::string m_whitespace;

    private:
        std::vector<std::string> m_jvmOptions;

};



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H