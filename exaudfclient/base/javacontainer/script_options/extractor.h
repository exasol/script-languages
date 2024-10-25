#ifndef SCRIPTOPTIONLINEPEXTRACTOR_H
#define SCRIPTOPTIONLINEPEXTRACTOR_H 1

#include <string>
#include <vector>
#include <functional>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

/**
 * Abstract interface for the Extractor class.
 * Defines methods to extract the Java options from the script code.
 * The extraction process searches for the known Java Options and handles them appropriatly.
 * The script code is then modified, where the found options are removed.
 * The interface defines methods to access the found Jar- and JvmOption options.
 * The scriptclass and import options are processed internally by the respective extractor implementation.
 */
struct Extractor {

    typedef std::function<void(const std::string &option)> tJarIteratorCallback;

    virtual ~Extractor() {}

    /**
     * Access the found Jar paths. For each found jar path the given callback function is called with the jar option as argument.
     */
    virtual void iterateJarPaths(tJarIteratorCallback callback) const = 0;

    /**
     * Access the Jvm option options. The extractor implementations must store the found Jvm Options in a std::vector.
     * The vector is returned as rvalue.
     */
    virtual std::vector<std::string>&& moveJvmOptions() = 0;

    /**
     * Run the extraction. This will:
     * 1. Add the first `scriptclass` option to the list of Jvm options.
     * 2. Replace and (nested) reference of an `import` script
     * 3. Find and store all Jar options
     * 4. Find and store all `jvmoption` options
     * 5. Remove `scriptclass`, `jar`, `import` and `jvmoption` from the script code. The behavior is implementation specific.
     */
    virtual void extract(std::string & scriptCode) = 0;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPEXTRACTOR_H