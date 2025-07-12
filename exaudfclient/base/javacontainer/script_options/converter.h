#ifndef SCRIPTOPTIONLINEPARSERCONVERTER_H
#define SCRIPTOPTIONLINEPARSERCONVERTER_H 1

#include <string>
#include <vector>
#include <functional>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

/**
 * This class retrieves the raw Java script option values (scriptclass, jvmoption, jar)
 * and converts them to the proper format expected by the JvmContainerImpl class.
 * Besides the converter functions it provides methods to access the converted values.
 */
class Converter {

    public:
        typedef std::function<void(const std::string &option)> tJarIteratorCallback;

        Converter();
    
        void convertScriptClassName(const std::string & value);

        virtual void convertJvmOption(const std::string & value) = 0;

        std::vector<std::string>&& moveJvmOptions() {
            return std::move(m_jvmOptions);
        }

        virtual void convertExternalJar(const std::string & value) = 0;

        virtual void iterateJarPaths(tJarIteratorCallback callback) const = 0;

    protected:

        std::vector<std::string> m_jvmOptions;

        const std::string m_whitespace;
};



} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCONVERTER_H