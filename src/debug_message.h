
#ifndef NDEBUG

#define DBGVAR( os, var ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << #var << " = [" << (var) << "]" << std::endl

#define DBGMSG( os, msg ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") "  \
       << msg << std::endl

#define DBG_FUNC_BEGIN( os ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") Begin function ["  \
       << __func__ << "]" << std::endl

#define DBG_FUNC_END( os ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") End function ["  \
       << __func__ << "]" << std::endl

#else

#define DBGVAR( os, var )

#define DBGMSG( os, msg )

#define DBG_FUNC_BEGIN( os )

#define DBG_FUNC_END( os )

#endif