#ifndef TEST_SYMBOL_SCANNER_H
#define TEST_SYMBOL_SCANNER_H

/**
 * Scan all loaded shared objects for search string.
 * Iterates over the link-map (Primary link-map; in other words the primary linker namespace) and searches
 * the shared object name matches on of the \t separated search strings.
 */
extern "C" {
    const char* scan_symbols(const char* search_strings);
}

#endif //TEST_SYMBOL_SCANNER_H
