#include "base/javacontainer/script_options/parser_ctpg_jvm_options_parser.h"
#include "base/script_options_parser/ctpg/ctpg.hpp"
#include <sstream>

using namespace exaudf_ctpg;
using namespace exaudf_ctpg::ftors;

namespace SWIGVMContainers {

namespace JavaScriptOptions {

namespace JvmOptionsCTPG {


const auto to_options(std::string&& e)
{
    tJvmOptions options{e};
    return options;
}

auto&& add_option(std::string&& e, tJvmOptions&& ob)
{
    ob.push_back(std::move(e));
    return std::move(ob);
}

auto convert_escape_sequence(std::string_view escape_seq)
{
    if (escape_seq == R"_(\ )_") {
        return std::string_view(" ");
    } else if (escape_seq == R"_(\t)_") {
        return std::string_view("\t");
    } else if (escape_seq == R"_(\f)_") {
        return std::string_view("\f");
    } else if (escape_seq == R"_(\v)_") {
        return std::string_view("\v");
    } else if (escape_seq == R"_(\\)_") {
        return std::string_view("\\");
    } else {
        throw std::invalid_argument(std::string("Internal parser error: Unexpected escape sequence " + std::string(escape_seq)));
    }
}


constexpr char not_separator_pattern[] = R"_([^ \x09\x0c\x0b])_";
constexpr char whitespaces_pattern[] = R"_([ \x09\x0c\x0b]+)_";
constexpr char escape_pattern[] = R"_(\\\\|\\t|\\ |\\f|\\v)_";

constexpr regex_term<not_separator_pattern> not_separator("not_separator");
constexpr regex_term<whitespaces_pattern> whitespaces("whitespace");
constexpr regex_term<escape_pattern> escape_seq("escape_seq");

constexpr nterm<tJvmOptions> text("text");
constexpr nterm<std::string> option_element("option_element");


constexpr parser jvm_option_parser(
    text,
    terms(escape_seq, not_separator, whitespaces),
    nterms(text, option_element),
    rules(
        text()
            >= [] () {return tJvmOptions();},
        text(option_element)
            >= [](auto&& e) { return to_options(std::move(e)); },
        text(text, whitespaces)
            >= [](auto&& ob, skip) { return std::move(ob); },
        text(text, whitespaces, option_element)
            >= [](auto&& ob, skip, auto&& e) { return add_option(std::move(e), std::move(ob)); },
        option_element(not_separator)
            >= [](auto ns) { return std::string(ns); },
        option_element(option_element, not_separator)
            >= [](auto &&ob, auto c) { return std::move(ob.append(c)); },
        option_element(option_element, escape_seq)
            >= [](auto &&ob, auto es) { return std::move(ob.append(convert_escape_sequence(es))); }
    )
);


void parseJvmOptions(const std::string & jvmOptions, tJvmOptions& result) {
    std::stringstream error_buffer;
    auto && res = jvm_option_parser.parse(
        parse_options{}.set_skip_whitespace(false),
        buffers::string_buffer(jvmOptions.c_str()),
        error_buffer);
    if (res.has_value()) {
        result.insert(result.end(), res.value().begin(), res.value().end());
    }
    else {
        throw std::invalid_argument(error_buffer.str());
    }
}

} //namespace JvmOptionsCTPG

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
