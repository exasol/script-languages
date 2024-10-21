#include "script_option_lines_ctpg.h"
#include "ctpg.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include "base/script_options_parser/exception.h"

using namespace exaudf_ctpg;
using namespace exaudf_ctpg::ftors;


namespace ExecutionGraph
{

namespace OptionsLineParser
{


namespace CTPG {


struct Option {
    std::string key;
    std::string value;
    source_point start;
    source_point end;
};

using options_type = std::vector<Option>;

namespace ParserInternals {

const auto empty_options()
{
    options_type ob;
    return ob;
}

const auto to_options(Option&& e)
{
    options_type ob{e};
    return ob;
}

const auto to_option(std::string_view key, std::string value, source_point sp_begin, source_point sp_end)
{
    return Option{std::string(key), value, sp_begin, sp_end};
}

auto&& add_option(Option&& e, options_type&& ob)
{
    ob.push_back(std::move(e));
    return std::move(ob);
}

const auto convert_escape_seq(std::string_view escape_seq) {
    std::string retVal;
    if (escape_seq == R"_(\;)_") {
        retVal = ";";
    } else if (escape_seq == R"_(\n)_") {
        retVal = "\n";
    } else if (escape_seq == R"_(\r)_") {
        retVal = "\r";
    } else {
        throw OptionParserException(std::string("Internal parser error: Unexpected escape sequence " + std::string(escape_seq)));
    }

    return retVal;
}


const auto convert_whitespace_escape_seq(std::string_view escape_seq) {
    std::string retVal;
    if (escape_seq == R"_(\ )_") {
        retVal = " ";
    } else if (escape_seq == R"_(\t)_") {
        retVal = "\t";
    } else if (escape_seq == R"_(\f)_") {
        retVal = "\f";
    } else if (escape_seq == R"_(\v)_") {
        retVal = "\v";
    } else {
        throw OptionParserException(std::string("Internal parser error: Unexpected white space escape sequence " + std::string(escape_seq)));
    }

    return retVal;
}



constexpr char alpha_numeric_pattern[] = R"_([0-9a-zA-Z_]+)_";
constexpr char not_semicolon_pattern[] = R"_([^;])_";
constexpr char whitespaces_pattern[] = R"_([ \x09\x0c\x0b]+)_";
constexpr char escape_pattern[] = R"_(\\;|\\n|\\r)_";
constexpr char whitespace_escape_pattern[] = R"_(\\ |\\t|\\f|\\v)_";



constexpr char_term start_option_token('%');
constexpr char_term end_option_token(';');
constexpr regex_term<alpha_numeric_pattern> alpha_numeric("alpha_numeric");
constexpr regex_term<not_semicolon_pattern> not_semicolon("not_semicolon");
constexpr regex_term<whitespaces_pattern> whitespaces("whitespace");
constexpr regex_term<escape_pattern> escape_seq("escape_seq");
constexpr regex_term<whitespace_escape_pattern> whitespace_escape_seq("escape_seq");

constexpr nterm<options_type> text("text");
constexpr nterm<options_type> options("options");
constexpr nterm<Option> option_element("option_element");
constexpr nterm<int> rest("rest");
constexpr nterm<std::string> option_value("option_value");


constexpr parser option_parser(
    text,
    terms(start_option_token, escape_seq, whitespace_escape_seq, whitespaces, end_option_token, alpha_numeric, not_semicolon),
    nterms(text, option_value, options, option_element, rest),
    rules(
        text(rest)
            >= [] (skip) {return empty_options();},
        text(options, rest)
            >= [] (auto o, skip) {return o;},
        text(options)
            >= [] (auto o) {return o;},
        options(option_element)
            >= [](auto&& e) { return to_options(std::move(e)); },
        options(options, option_element)
            >= [](auto&& ob, auto&& e) { return add_option(std::move(e), std::move(ob)); },
        option_element(start_option_token, alpha_numeric, whitespaces, option_value, end_option_token)
            >= [](auto st, auto ok, skip, auto ov, auto e) { return to_option(ok.get_value(), ov, st.get_sp(), e.get_sp()); },
        option_element(whitespaces, start_option_token, alpha_numeric, whitespaces, option_value, end_option_token)
            >= [](skip, auto st, auto ok, skip, auto ov, auto e) { return to_option(ok.get_value(), ov, st.get_sp(), e.get_sp()); },
        option_value(alpha_numeric)
            >= [](auto o) { return std::string(o.get_value()); },
        option_value(not_semicolon)
            >= [](auto o) { return std::string(o.get_value()); },
        option_value(whitespace_escape_seq)
            >= [](auto o) { return std::string(convert_whitespace_escape_seq(o.get_value())); },
        option_value(escape_seq)
            >= [](auto es) { return convert_escape_seq(es.get_value()); },
        option_value(option_value, not_semicolon)
            >= [](auto&& ov, auto v) { return std::move(ov.append(v.get_value())); },
        option_value(option_value, whitespace_escape_seq)
            >= [](auto&& ov, auto es) { return std::move(ov.append(es.get_value())); },
        option_value(option_value, escape_seq)
            >= [](auto&& ov, auto es) { return std::move(ov.append(convert_escape_seq(es.get_value()))); },
        option_value(option_value, start_option_token)
            >= [](auto&& ov, auto v) { return std::move(ov.append("%")); },
        option_value(option_value, alpha_numeric)
            >= [](auto&& ov, auto v) { return std::move(ov.append(v.get_value())); },
        option_value(option_value, whitespaces)
            >= [](auto&& ov, auto v) { return std::move(ov.append(v.get_value())); },
        rest(alpha_numeric)
            >= [](auto r) { return 0;},
        rest(whitespaces)
            >= [](auto r) { return 0;},
        rest(escape_seq)
            >= [](auto r) { return 0;},
        rest(whitespace_escape_seq)
            >= [](auto r) { return 0;},
        rest(end_option_token)
            >= [](auto r) { return 0;},
        rest(not_semicolon)
            >= [](auto r) { return 0;},
        rest(rest, alpha_numeric)
            >= [](auto r, skip) { return 0;},
        rest(rest, not_semicolon)
            >= [](auto r, skip) { return 0;},
        rest(rest, whitespaces)
            >= [](auto r, skip) { return 0;},
        rest(rest, end_option_token)
            >= [](auto r, skip) { return 0;},
        rest(rest, start_option_token)
            >= [](auto r, skip) { return 0;},
        rest(rest, escape_seq)
            >= [](auto r, skip) { return 0;},
        rest(rest, whitespace_escape_seq)
            >= [](auto r, skip) { return 0;}
    )
);

void parse(std::string&& code, options_type& result) {

    std::stringstream error_buffer;
    auto res = option_parser.parse(
        parse_options{}.set_skip_whitespace(false),
        buffers::string_buffer(std::move(code)),
        error_buffer);
    if (res.has_value()) {
        result = res.value();
    }
    else {
        std::stringstream ss;
        ss << "Error parsing script options: " << error_buffer.str();
        throw OptionParserException(ss.str());
    }
}

} //namespace ParserInternals

struct LinePositions {
    size_t mStartPos;
    size_t mEndPos;
};

inline std::optional<LinePositions> getNextLine(const size_t current_pos, const std::string & scriptCode) {
    /**
     * Find first of occurence of '%', starting search from position 'current_pos'.
     * If no '%' is found, return an empty result.
     * If '%' is found, search backwards from '%' for '\n' or \r':
     *  1. If not found, '%' was found in the first line. Then we can set 'new_option_start_pos'=0
     *  2. If found, set new_option_start_pos to position 1 char behind pos of found '\n' or '\r'.
     * Then search forward for next occurence of '\n' or \r' and assign to var 'line_end_pos':
        1. If not found, 'line_end_pos' will get assigned std::string::npos (std::string::substr(...,npos), returns substring until end of string
        2. If found, 'line_end_pos' will assigned to position of line end of line where '%' was found
     */
    std::optional<LinePositions> retVal;
    const size_t new_option_start_pos = scriptCode.find_first_of("%", current_pos);
    if (new_option_start_pos == std::string::npos) {
        return retVal;
    }
    size_t line_start_pos = scriptCode.find_last_of("\r\n", new_option_start_pos);
    if (std::string::npos == line_start_pos) {
        line_start_pos = 0;
    }
    else {
        line_start_pos++;
    }

    const size_t line_end_pos = scriptCode.find_first_of("\r\n", line_start_pos);
    retVal = LinePositions{ .mStartPos = line_start_pos, .mEndPos = line_end_pos};
    return retVal;
}

void parseOptions(const std::string& code, options_map_t & result) {

    size_t current_pos = 0;
    std::optional<LinePositions> currentLinePositions = getNextLine(current_pos, code);
    while (currentLinePositions) {

        std::string line = code.substr(currentLinePositions->mStartPos, currentLinePositions->mEndPos);
        options_type parser_result;
        ParserInternals::parse(std::move(line), parser_result);
        for (const auto & option: parser_result) {
            ScriptOption entry = {
                .value = option.value,
                .idx_in_source = currentLinePositions->mStartPos + option.start.column - 1,
                .size = option.end.column - option.start.column + 1
            };
            auto it_in_result = result.find(option.key);
            if (it_in_result == result.end()) {
                options_t new_options;
                new_options.push_back(entry);
                result.insert(std::make_pair(option.key, new_options));
            }
            else {
                it_in_result->second.push_back(entry);
            }
        }
        if (currentLinePositions->mEndPos == std::string::npos) {
            break;
        }
        current_pos =  currentLinePositions->mEndPos + 1;

        currentLinePositions = getNextLine(current_pos, code);
    }
}


} // namespace CTPG

} // namespace OptionsLineParser

} // namespace ExecutionGraph