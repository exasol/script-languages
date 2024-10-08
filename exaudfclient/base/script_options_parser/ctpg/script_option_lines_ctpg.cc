#include "script_option_lines_ctpg.h"
#include "ctpg.hpp"
#include <iostream>
#include <string>
#include <sstream>

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



constexpr char alpha_numeric_pattern[] = R"_([0-9a-zA-Z_]+)_";
constexpr char not_semicolon_pattern[] = R"_([^;])_";
constexpr char whitespaces_pattern[] = R"_([ \x09\x0c\x0b]+)_";


constexpr char_term start_option_token('%');
constexpr char_term end_option_token(';');
constexpr regex_term<alpha_numeric_pattern> alpha_numeric("alpha_numeric");
constexpr regex_term<not_semicolon_pattern> not_semicolon("not_semicolon");
constexpr regex_term<whitespaces_pattern> whitespaces("whitespace");
constexpr string_term semicolon_escape(R"_(\;)_");

constexpr nterm<options_type> text("text");
constexpr nterm<options_type> options("options");
constexpr nterm<Option> option_element("option_element");
constexpr nterm<int> rest("rest");
constexpr nterm<std::string> option_value("option_value");


constexpr parser option_parser(
    text,
    terms(start_option_token, semicolon_escape, whitespaces, end_option_token, alpha_numeric, not_semicolon),
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
        option_value(whitespaces)
            >= [](auto o) { return std::string(o.get_value()); },
        option_value(semicolon_escape)
            >= [](auto o) { return std::string(";"); },
        option_value(option_value, not_semicolon)
            >= [](auto&& ov, auto v) { return std::move(ov.append(v.get_value())); },
        option_value(option_value, semicolon_escape)
            >= [](auto&& ov, auto v) { return std::move(ov.append(";")); },
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
        rest(semicolon_escape)
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
            >= [](auto r, skip) { return 0;}
    )
);

void parse(std::string&& code, options_type& result, std::function<void(const char*)> throwException) {
    std::stringstream error_buffer;
    auto res = option_parser.parse(
        parse_options{}.set_skip_whitespace(false),
        buffers::string_buffer(std::move(code)),
        error_buffer);
    if (res.has_value())
    {
        result = res.value();
    }
    else
    {
        std::stringstream ss;
        ss << "Error parsing script options: " << error_buffer.str();
        throwException(ss.str().c_str());
    }
}

} //namespace ParserInternals

void parseOptions(const std::string& code, options_map_t & result, std::function<void(const char*)> throwException) {

    size_t current_pos = 0;

    do {

        const size_t new_pos = code.find_first_of("\r\n", current_pos);
        std::string line = code.substr(current_pos, new_pos);
        if (!line.empty() && !std::all_of(line.begin(),line.end(), [](const char c) {return std::isspace(c);})) {
            options_type parser_result;
            ParserInternals::parse(std::move(line), parser_result, throwException);
            for (const auto & option: parser_result)
            {
                ScriptOption entry = {
                    .value = option.value,
                    .idx_in_source = current_pos + option.start.column - 1,
                    .size = option.end.column - option.start.column + 1
                };
                auto it_in_result = result.find(option.key);
                if (it_in_result == result.end())
                {
                    options_t new_options;
                    new_options.push_back(entry);
                    result.insert(std::make_pair(option.key, new_options));
                }
                else
                {
                    it_in_result->second.push_back(entry);
                }
            }
        }
        if (new_pos == std::string::npos) {
            break;
        }
        current_pos =  new_pos + 1;
    } while(true);
}


} // namespace CTPG

} // namespace OptionsLineParser

} // namespace ExecutionGraph