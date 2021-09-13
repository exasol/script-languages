#include "symbol_scanner.h"
#include <link.h>
#include <string>
#include <vector>

namespace SymbolScanner {
static std::string gResult;
static std::vector<std::string> gSearchStrings;

static const char delimeter = '\t';
}

using namespace SymbolScanner;

inline void split_search_strings(const char* search_strings) {
    size_t start;
    size_t end = 0;

    std::string str(search_strings);
    printf("Split:'%s'\n", search_strings);
    while ((start = str.find_first_not_of(delimeter, end)) != std::string::npos) {
        end = str.find(delimeter, start);
        gSearchStrings.push_back(str.substr(start, end - start));
    }

    printf("Found:%zu search strings\n", gSearchStrings.size());
}

static int callback(struct dl_phdr_info *info, size_t size, void *data)
{
    const std::string dlpiName(info->dlpi_name);
    for (auto searchString : gSearchStrings) {
        printf("Searching for %s in %s\n", searchString.c_str(), dlpiName.c_str());
        if (dlpiName.find(searchString) != std::string::npos) {
            gResult.append(dlpiName + delimeter);
        }
    }
    return 0;
}

extern "C" {
    const char* scan_symbols(const char* search_strings) {
        split_search_strings(search_strings);
        dl_iterate_phdr(callback, NULL);
        return gResult.c_str();
    }
}
