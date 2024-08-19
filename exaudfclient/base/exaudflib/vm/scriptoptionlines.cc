#include "scriptoptionlines.h"
#include "ctpg.hpp"
#include <string>
#include <sstream>



namespace ExecutionGraph
{

namespace Parser {

using namespace exaudf_ctpg;
using namespace exaudf_ctpg::ftors;


struct Option {
    std::string o;
    source_point start;
    source_point end;
};

using options_type = std::vector<Option>;

auto empty_options()
{
    options_type ob;
    return ob;
}

auto to_options(Option&& e)
{
    options_type ob{e};
    return ob;
}

auto to_option(std::string_view key, source_point sp_begin, source_point sp_end)
{
    std::string s_key = std::string(key);
    return Option{std::move(s_key), sp_begin, sp_end};
}

auto&& add_option(Option&& e, options_type&& ob)
{
    ob.push_back(std::move(e));
    return std::move(ob);
}


auto convert_escape_sequence(std::string_view& v)
{
    std::string s_v = std::string(v);
    return s_v.substr(1, 1);
}


constexpr char option_pattern[] = R"_([0-9a-zA-Z_\t "']+)_";
constexpr char everything_else_pattern[] = R"_([^0-9a-zA-Z_\t ;%\\"']+)_";
constexpr char not_an_option_start_pattern[] = R"_([^%])_";
constexpr char escaped_sequence_pattern[] = R"_((\\%|\\;))_";

constexpr char_term start_option_tag('%');
constexpr char_term end_option_tag(';');
constexpr regex_term<option_pattern> option("option");
constexpr regex_term<everything_else_pattern> everything_else("everything_else");
constexpr regex_term<not_an_option_start_pattern> not_an_option_start("not_an_option_start");
constexpr regex_term<escaped_sequence_pattern> escaped_sequence("escaped_sequence");


constexpr nterm<options_type> text("text");
constexpr nterm<options_type> options("options");
constexpr nterm<Option> option_element("option_element");
constexpr nterm<int> rest("rest");
constexpr nterm<std::string> option_value("option_value");


constexpr parser option_parser(
    text,
    terms(start_option_tag, end_option_tag, option, escaped_sequence, everything_else, not_an_option_start),
    nterms(text, options, option_element, option_value, rest),
    rules(
        text(options, rest)
            >= [] (auto o, skip) {return o;},
        options()
            >= [] () {return empty_options();},
        options(option_element)
            >= [](auto&& e) { return to_options(std::move(e)); },
        options(options, option_element)
            >= [](auto&& ob, auto&& e) { return add_option(std::move(e), std::move(ob)); },
        option_element(start_option_tag, option_value, end_option_tag)
            >= [](auto s, auto o, auto e) { return to_option(o, s.get_sp(), e.get_sp()); },
        option_value(option)
            >= [](auto o) { return std::string(o.get_value()); },
        option_value(escaped_sequence)
            >= [](auto o) { return convert_escape_sequence(o.get_value()); },
        option_value(option_value, option)
            >= [](auto ov, auto v) { return ov.append(v.get_value()); },
        option_value(option_value, escaped_sequence)
            >= [](auto ov, auto v) { return ov.append(convert_escape_sequence(v.get_value())); },
        rest(rest, option)
            >= [](auto r, skip) { return 0;},
        rest(rest, everything_else)
            >= [](auto r, skip) { return 0;},
        rest(rest, start_option_tag)
            >= [](auto r, skip) { return 0;},
        rest(rest, end_option_tag)
            >= [](auto r, skip) { return 0;},
        rest(rest, escaped_sequence)
            >= [](auto r, skip) { return 0;},
        rest(rest, not_an_option_start)
            >= [](auto r, skip) { return 0;},
        rest()
            >=  []() { return 0;}
    )
);

std::string extractOptionLine(std::string& code, const std::string option, size_t& pos, std::function<void(const char*)>throwException)
{
    std::string ret_val;
    std::stringstream error_stream;
    auto res = Parser::option_parser.parse(
        //parse_options{}.set_verbose(),
        buffers::string_buffer(Parser.c_str()),
        error_stream);
    if (res.has_value())
    {
        int (*IsSpace)(int) = std::isspace;
        for (const auto& w : res.value())
        {
            const auto key_end_index = std::find_if(w.o.begin(), w.o.end(), IsSpace);
            if ( key_end_index != w.o.end()) {
                const std::string key = w.o(w.o.begin(), key_end_index);
                if (key == option) {
                    ret_val = std::string(key_end_index, w.o.end());
                    pos = w.o.start.global;
                }
            }
            else {
                error_stream << "Option '" << w.o << "' has invalid syntax";
                throwException(error_stream.str().c_str());
            }
        }
    } else {
        throwException(error_stream.str().c_str());
    }
    return ret_val;
}

} // namespace ExecutionGraph
