#include "base/javacontainer/script_options/converter_v2.h"
#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

ConverterV2::ConverterV2()
: Converter()
, m_jarPaths() {}

void ConverterV2::convertExternalJar(const std::string & value) {
    std::istringstream stream(value);
    std::string jar;

    while (std::getline(stream, jar, ':')) {
        m_jarPaths.push_back(jar);
    }
}

class ProcessJvmOptionsStatemachine {
    public:
        typedef std::function<void(const std::string &jvmoption)> tOptionFoundCallback;
        ProcessJvmOptionsStatemachine(tOptionFoundCallback cb, const std::string &whitespaces);
    
        void processCharacter(const char current_char);
        
        void finish();

    private:
        void processWhitespace (const char current_char);
        void processNormalCharacter(const char current_char);
        void processBackslash(const char current_char);
        enum ParserState {
            START = 0,
            IN_VALUE = 1,
            IN_ESCAPE = 2,
            IN_WHITESPACE = 3
        };

        ParserState m_state;

        tOptionFoundCallback m_optionsFoundCallback;
        
        const std::string &m_whitespaces;
        
        std::string m_currentOption;
};

ProcessJvmOptionsStatemachine::ProcessJvmOptionsStatemachine(tOptionFoundCallback cb, const std::string &whitespaces)
: m_state(START)
, m_optionsFoundCallback(cb) 
, m_whitespaces(whitespaces)
, m_currentOption() {}

void ProcessJvmOptionsStatemachine::processCharacter(const char current_char) {
    if (current_char == '\\') {
        processBackslash(current_char);
    }
    else if (m_whitespaces.find(current_char) != std::string::npos) {
        processWhitespace(current_char);
    }
    else {
        processNormalCharacter(current_char);
    }
}


void ProcessJvmOptionsStatemachine::processWhitespace (const char current_char) {
    switch (m_state) {
        case START:
            break;
        case IN_VALUE:
            m_state = IN_WHITESPACE;
            m_optionsFoundCallback(m_currentOption);
            m_currentOption.clear();
            break;
        case IN_ESCAPE:
            if (current_char == ' ') {
                m_currentOption.push_back(current_char);
                m_state = IN_VALUE;
            } else {
                std::stringstream ss;
                ss << "Unexpected escape sequence. Invalid whitespace character '" << current_char << "'";
                throw std::invalid_argument(ss.str());
            }
            break;
        case IN_WHITESPACE:
            break;
        }
}

void ProcessJvmOptionsStatemachine::processNormalCharacter(const char current_char) {
    switch (m_state) {
        case START:
            m_currentOption.push_back(current_char);
            m_state = IN_VALUE;
            break;
        case IN_VALUE:
            m_currentOption.push_back(current_char);
            break;
        case IN_ESCAPE:
            if (current_char == 't') {
                m_currentOption.push_back('\t');
                m_state = IN_VALUE;
            } else if (current_char == 'f') {
                m_currentOption.push_back('\f');
                m_state = IN_VALUE;
            } else if (current_char == 'v') {
                m_currentOption.push_back('\v');
                m_state = IN_VALUE;
            } else {
                std::stringstream ss;
                ss << "Invalid escape sequence at character '" << current_char << "'";
                throw std::invalid_argument(ss.str());
            }
            break;
        case IN_WHITESPACE:
            m_currentOption.push_back(current_char);
            m_state = IN_VALUE;
            break;
        }
}

void ProcessJvmOptionsStatemachine::processBackslash(const char current_char) {
    switch (m_state) {
        case START:
        case IN_VALUE:
        case IN_WHITESPACE:
            m_state = IN_ESCAPE;
            break;
        case IN_ESCAPE:
            m_currentOption.push_back(current_char);
            m_state = IN_VALUE;
            break;
        }
}

void ProcessJvmOptionsStatemachine::finish() {
    if (m_currentOption.size() > 0) {
        m_optionsFoundCallback(m_currentOption);
    }
}


void ConverterV2::convertJvmOption(const std::string & value) {
    auto optionFoundCallback = [&] (const std::string & jvmoption) {m_jvmOptions.push_back(jvmoption);};
    ProcessJvmOptionsStatemachine sm(optionFoundCallback, m_whitespace);
    try {
        std::for_each(value.cbegin(), value.cend(), [&sm](const char c) {sm.processCharacter(c);});
    } catch(const std::invalid_argument & e) {
        std::stringstream ss;
        ss << "F-UDF-CL-SL-JAVA-1629 " << "Error parsing jvmoption: " << value << " - " << e.what();
        throw std::invalid_argument(ss.str());
    }
      sm.finish();
}

void ConverterV2::iterateJarPaths(Converter::tJarIteratorCallback callback) const {
    std::for_each(m_jarPaths.begin(), m_jarPaths.end(), callback);
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
