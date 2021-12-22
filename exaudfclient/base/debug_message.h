#ifndef DEBUG_MESSAGE_H
#define DEBUG_MESSAGE_H

#include "date.h"


inline std::string time_stamp() {
        using namespace std::chrono;
        auto now = time_point_cast<milliseconds>(system_clock::now());
        return date::format("%T", now);
}

#define PRINT_EXCEPTION( os, error_code, ex ) \
  (os) << (error_code) << ":" << " Caught Exception: " << (ex.what()) << std::endl

#define PRINT_ERROR_MESSAGE( os, error_code, error_message ) \
  (os) << (error_code) << ":" << error_message << std::endl

#ifndef NDEBUG

#define DBG_PROFILE( os, msg) \
    (os) << "PROFILING[" << msg << "] " << time_stamp() << std::endl

#define DBG_EXCEPTION( os, ex ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") in " << __func__ << " " \
       << "EXCEPTION: " << #ex << " = [" << (ex.what()) << "]" << std::endl

#define DBGVAR( os, var ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") in " << __func__ << " " \
       << "VAR: " << #var << " = [" << (var) << "]" << std::endl

#define DBGMSG( os, msg ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") in " << __func__ << " " \
       << (msg) << std::endl

#define DBG_STREAM_MSG( os, msg ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") in " << __func__ << " " \
       << msg << std::endl


#define DBG_FUNC_BEGIN( os ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") BEGIN FUNC: "  \
       << __func__ << std::endl

#define DBG_FUNC_END( os ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") END FUNC: "  \
       << __func__  << std::endl

#define DBG_FUNC_CALL( os, call ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << "CALL BEGIN: " << #call << std::endl; \
  call; \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << "CALL END: " << #call << std::endl 

#define DBG_COND_FUNC_CALL( os, call ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << "CALL BEGIN: " << #call << std::endl; \
  call; \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << "CALL END: " << #call << std::endl 

#else

#define DBG_EXCEPTION( os, ex )

#define DBGVAR( os, var )

#define DBGMSG( os, msg )

#define DBG_STREAM_MSG( os, msg )

#define DBG_FUNC_BEGIN( os )

#define DBG_FUNC_END( os )

#define DBG_FUNC_CALL( os, call ) call

#define DBG_COND_FUNC_CALL( os, call )

#define DBG_PROFILE( os, msg) \
    (os) << "PROFILING[" << msg << "] " << time_stamp() << std::endl

#endif

#endif
