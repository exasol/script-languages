#include <string>
#include <cstring>

#include "exa_parser_cfg.h"

//  Parser option 2 means use ctpg parser.
//  argv_parser_option is command line argument as it is
bool is_use_ctpg_parser(const std::string& argv_parser_option) {
    bool use_ctpg_option_parser = false;

    //  env var has higher priority than argv value.
    const char* env_val = ::getenv("SCRIPT_OPTIONS_PARSER_VERSION");
    if(env_val) {
        use_ctpg_option_parser = (strcmp(env_val, "2") == 0);
    }
    else if(argv_parser_option.compare("scriptOptionsParserVersion=2") == 0) {
        use_ctpg_option_parser = true;
    }
    return use_ctpg_option_parser;    
}