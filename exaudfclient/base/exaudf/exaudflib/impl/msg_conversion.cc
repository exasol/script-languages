#include "exaudf/exaudflib/impl/msg_conversion.h"

std::string exaudflib::msg_conversion::convert_message_type_to_string(int message_type){
    switch (message_type)
    {
        case 0:
            return "MT_UNKNOWN";
        case 1:
            return "MT_CLIENT";
        case 2:
            return "MT_INFO";
        case 3:
            return "MT_META";
        case 4:
            return "MT_CLOSE";
        case 5:
            return "MT_IMPORT";
        case 6:
            return "MT_NEXT";
        case 7:
            return "MT_RESET";
        case 8:
            return "MT_EMIT";
        case 9:
            return "MT_RUN";
        case 10:
            return "MT_DONE";
        case 11:
            return "MT_CLEANUP";
        case 12:
            return "MT_FINISHED";
        case 13:
            return "MT_PING_PONG";
        case 14:
            return "MT_TRY_AGAIN";
        case 15:
            return "MT_CALL";
        case 16:
            return "MT_RETURN";
        case 17:
            return "MT_UNDEFINED_CALL";
        default:
            return "unknown: " + message_type;
    }
}

std::string exaudflib::msg_conversion::convert_type_to_string(int type){
    switch (type)
    {
        case 0:
            return "PB_UNSUPPORTED";
        case 1:
            return "PB_DOUBLE";
        case 2:
            return "PB_INT32";
        case 3:
            return "PB_INT64";
        case 4:
            return "PB_NUMERIC";
        case 5:
            return "PB_TIMESTAMP";
        case 6:
            return "PB_DATE";
        case 7:
            return "PB_STRING";
        case 8:
            return "PB_BOOLEAN";
        default:
            return "unknown: " + type;
    }
}
