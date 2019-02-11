
#ifndef NDEBUG

#define DBG_EXCEPTION( os, ex ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") in function " << __func__ << " " \
       << "EXCEPTION: " << #ex << " = [" << (ex.what()) << "]" << std::endl

#define DBGVAR( os, var ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << "VAR: " << #var << " = [" << (var) << "]" << std::endl

#define DBGMSG( os, msg ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") "  \
       << msg << std::endl

#define DBG_FUNC_BEGIN( os ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") Begin function ["  \
       << __func__ << "]" << std::endl

#define DBG_FUNC_END( os ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") End function ["  \
       << __func__ << "]" << std::endl

#define DBG_FUNC_CALL( os, call ) \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << "CALL BEGIN: " << #call << std::endl; \
  call; \
  (os) << "DBG: " << __FILE__ << "(" << __LINE__ << ") " \
       << "CALL END: " << #call << std::endl 
#else

#define DBG_EXCEPTION( os, ex )

#define DBGVAR( os, var )

#define DBGMSG( os, msg )

#define DBG_FUNC_BEGIN( os )

#define DBG_FUNC_END( os )

#define DBG_FUNC_CALL( os, call )

#endif