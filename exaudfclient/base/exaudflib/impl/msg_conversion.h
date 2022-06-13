#ifndef EXAUDFCLIENT_EXAUDFLIB_MSG_CONVERSION_H
#define EXAUDFCLIENT_EXAUDFLIB_MSG_CONVERSION_H

#include <string>

namespace exaudflib {
    namespace msg_conversion {
        std::string convert_message_type_to_string(int message_type);
        std::string convert_type_to_string(int type);
    }
}


#endif //EXAUDFCLIENT_EXAUDFLIB_MSG_CONVERSION_H
