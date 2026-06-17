#ifndef EXAUDFLIB_LOAD_DYNAMIC
#define EXAUDFLIB_LOAD_DYNAMIC

void set_exaudflib_handle(void* handle);
void* load_dynamic(const char* name);

#endif //EXAUDFLIB_LOAD_DYNAMIC