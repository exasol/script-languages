#include <exaudflib.h>
#include <R.h>
#include <Rdefines.h>
#include <Rembedded.h>
#include <Rinterface.h>
#include <R_ext/Parse.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Boolean.h>
#include <exascript_r.h>
#include <exascript_r_int.h>

using namespace SWIGVMContainers;
using namespace std;

extern "C" void R_init_exascript_r(void *dll);

class SWIGVMContainers::RVMImpl {
public:
    RVMImpl(bool checkOnly);
    ~RVMImpl() {}
    void shutdown();
    bool run();
    const char* singleCall(single_call_function_id_e fn,  const ExecutionGraph::ScriptDTO& args, string& calledUndefinedSingleCall);
private:
    bool m_checkOnly;
};

RVM::RVM(bool checkOnly) {
    try {
        m_impl = new RVMImpl(checkOnly);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }
}
bool RVM::run() {
    try {
        return m_impl->run();
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }
    return false;
}

const char* RVM::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) {
    try {
        return m_impl->singleCall(fn,args,calledUndefinedSingleCall);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }
    return strdup("<this is an error>");
}

void RVM::shutdown() {
    try {
        m_impl->shutdown();
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }
}

static void evaluate_code(const char *code) {
    SEXP cmdSexp, cmdexpr;
    ParseStatus status;
    PROTECT(cmdSexp = allocVector(STRSXP, 1));
    SET_STRING_ELT(cmdSexp, 0, mkChar(code));
    cmdexpr = PROTECT(R_ParseVector(cmdSexp, -1, &status, R_NilValue));
    if (status != PARSE_OK) {
        UNPROTECT(2);
        throw RVM::exception("Failed to parse code");
    }
    for (R_len_t i = 0; i < length(cmdexpr); i++) {
        int errorOccurred;
        R_tryEvalSilent(VECTOR_ELT(cmdexpr, i), R_GlobalEnv, &errorOccurred);
        if (errorOccurred) {
            UNPROTECT(2);
            const char *buf = R_curErrorBuf();
            throw RVM::exception(buf);
        }
    }
    UNPROTECT(2);
}

static void evaluate_code_protected(const char *code) {
    SEXP expr, cmd, fun;
    int errorOccurred;
    PROTECT(fun = findFun(install("INTERNAL_PARSE_WRAPPER__"), R_GlobalEnv));
    PROTECT(expr = allocVector(LANGSXP, 2));
    PROTECT(cmd = allocVector(STRSXP, 1));
    SET_STRING_ELT(cmd, 0, mkChar(code));
    SETCAR(expr, fun);
    SETCADR(expr, cmd);
    R_tryEvalSilent(expr, R_GlobalEnv, &errorOccurred);
    UNPROTECT(3);
    if (errorOccurred)
        throw RVM::exception(R_curErrorBuf());
}

#define RVM_next_block_gen(type, vtype, rtype, var, value, null)  \
static void RVM_next_block_set_##type(SWIGTableIterator *data, SEXP &col, int c, unsigned long r) { \
    vtype var = data->get##type(c); \
    rtype (col)[r] = data->wasNull() ? null : value;       \
}
RVM_next_block_gen(Int32, int32_t, INTEGER, t, t, NA_INTEGER)
RVM_next_block_gen(Int64, double, REAL, t, double(t), NA_REAL)
RVM_next_block_gen(Double, double, REAL, t, t, NA_REAL)
RVM_next_block_gen(Numeric, const char*, REAL, t, ::atof(t), NA_REAL)
RVM_next_block_gen(Boolean, bool, LOGICAL, t, int(t), NA_LOGICAL)
#undef RVM_next_block_gen

#define RVM_next_block_gen(type) \
static void RVM_next_block_set_##type(SWIGTableIterator *data, SEXP &col, int c, unsigned long r) { \
    const char *t = data->get##type(c);                                       \
    if (!data->wasNull()) SET_STRING_ELT(col, r, mkChar(t)); \
    else SET_STRING_ELT(col, r, NA_STRING); \
}
RVM_next_block_gen(String)
RVM_next_block_gen(Date)
RVM_next_block_gen(Timestamp)
#undef RVM_next_block_gen

typedef void (*r_set_fun_t)(SWIGTableIterator *, SEXP &, int, unsigned long);

static void RVM_emit_block_set_Int32(SWIGResultHandler *data, SEXP &col, long c, long r)
{ if (INTEGER(col)[r] == NA_INTEGER) data->setNull(c); else data->setInt32(c, INTEGER(col)[r]); }
static void RVM_emit_block_set_Int64(SWIGResultHandler *data, SEXP &col, long c, long r)
{
    if (REAL(col)[r] == NA_REAL || ISNAN(REAL(col)[r]) || !R_FINITE(REAL(col)[r]))
        data->setNull(c);
    else data->setInt64(c, (int64_t)REAL(col)[r]);
}
static void RVM_emit_block_set_Double(SWIGResultHandler *data, SEXP &col, long c, long r)
{
    if (REAL(col)[r] == NA_REAL || ISNAN(REAL(col)[r]) || !R_FINITE(REAL(col)[r]))
        data->setNull(c);
    else data->setDouble(c, REAL(col)[r]);
}
static void RVM_emit_block_set_Boolean(SWIGResultHandler *data, SEXP &col, long c, long r)
{ if (LOGICAL(col)[r] == NA_LOGICAL) data->setNull(c); else data->setBoolean(c, LOGICAL(col)[r]); }
static void RVM_emit_block_set_Numeric(SWIGResultHandler *data, SEXP &col, long c, long r)
{ if (STRING_ELT(col, r) == NA_STRING) data->setNull(c); else data->setNumeric(c, CHAR(STRING_ELT(col, r))); }
static void RVM_emit_block_set_Date(SWIGResultHandler *data, SEXP &col, long c, long r)
{ if (STRING_ELT(col, r) == NA_STRING) data->setNull(c); else data->setDate(c, CHAR(STRING_ELT(col, r))); }
static void RVM_emit_block_set_Timestamp(SWIGResultHandler *data, SEXP &col, long c, long r)
{ if (STRING_ELT(col, r) == NA_STRING) data->setNull(c); else data->setTimestamp(c, CHAR(STRING_ELT(col, r))); }
static void RVM_emit_block_set_String(SWIGResultHandler *data, SEXP &col, long c, long r)
{
    if (STRING_ELT(col, r) == NA_STRING) data->setNull(c);
    else {
        const std::string s(CHAR(STRING_ELT(col, r)));
        data->setString(c, s.c_str(), s.size());
    }
}

extern "C" {
    SEXP RVM_next_block(SEXP dataexp, SEXP rowstofetch, SEXP buffer, SEXP buffersize) {
        SWIGTableIterator *data = reinterpret_cast<SWIGTableIterator*>(R_ExternalPtrAddr(dataexp));
        std::vector<std::string> &colnames = *(SWIGVM_params->inp_names);
        std::vector<SWIGVM_columntype_t> &coltypes = *(SWIGVM_params->inp_types);
        long cols_count = colnames.size();
        long rows = data->rowsInGroup() - data->rowsCompleted() + 1;
        long rowsalloc = rows;
        unsigned long currow = 0;
        SEXP cols[cols_count], ret = NULL, retnames;
        r_set_fun_t setfs[cols_count];
        bool ret_is_buffer = false;
        
        if (INTEGER(rowstofetch)[0] != NA_INTEGER) {
            if (INTEGER(rowstofetch)[0] < 0)
            {
                rowsalloc = rows = 1;
            }
            else if (INTEGER(rowstofetch)[0] < rows)
            {
                rowsalloc = rows = INTEGER(rowstofetch)[0];
            }
        }


        if (rowsalloc == 0)
        {
            rowsalloc = 1;
        }

        if (data->eot())
        {
            return R_NilValue;
        }

        if (INTEGER(buffersize)[0] == rowsalloc) {
            ret = buffer;
            ret_is_buffer = true;
            for (long c = 0; c < cols_count; ++c)
            {
                cols[c] = VECTOR_ELT(buffer, c);
            }
        }


        for (long c = 0; c < cols_count; ++c) {
            switch (coltypes[c].type) {
            case INT32:
                setfs[c] = &RVM_next_block_set_Int32;
                if (!ret_is_buffer) PROTECT(cols[c] = NEW_INTEGER(rowsalloc));
                break;
            case DOUBLE:
                setfs[c] = &RVM_next_block_set_Double;
                if (!ret_is_buffer) PROTECT(cols[c] = NEW_NUMERIC(rowsalloc));
                break;
            case INT64:
                setfs[c] = &RVM_next_block_set_Int64;
                if (!ret_is_buffer) PROTECT(cols[c] = NEW_NUMERIC(rowsalloc));
                break;
            case NUMERIC:
                setfs[c] = &RVM_next_block_set_Numeric;
                if (!ret_is_buffer) PROTECT(cols[c] = NEW_NUMERIC(rowsalloc));
                break;
            case STRING:
                setfs[c] = &RVM_next_block_set_String;
                if (!ret_is_buffer) PROTECT(cols[c] = NEW_CHARACTER(rowsalloc));
                break;
            case DATE:
                setfs[c] = &RVM_next_block_set_Date;
                if (!ret_is_buffer) PROTECT(cols[c] = NEW_CHARACTER(rowsalloc));
                break;
            case TIMESTAMP:
                setfs[c] = &RVM_next_block_set_Timestamp;
                if (!ret_is_buffer) PROTECT(cols[c] = NEW_CHARACTER(rowsalloc));
                break;
            case BOOLEAN:
                setfs[c] = &RVM_next_block_set_Boolean;
                if (!ret_is_buffer) PROTECT(cols[c] = NEW_LOGICAL(rowsalloc));
                break;
            default:
                SWIGVM_params->exch->setException("Internal error: wrong column type");
                break;
            }
        }


        if (rows > 0) {
            do {
                for (long c = 0; c < cols_count; ++c)
                {
                    setfs[c](data, cols[c], c, currow);
                }
                if (SWIGVM_params->exch->exthrowed) break;
                ++currow;
                if (!data->next()) break;
            } while(currow < (unsigned long)rows);
            if (!SWIGVM_params->exch->exthrowed && currow != (unsigned long)rows)
                SWIGVM_params->exch->setException("Could not read all rows");
        } else if (rows == 0) {
            for (long c = 0; c < cols_count; ++c)
                setfs[c](data, cols[c], c, currow);
        }



        if (!ret_is_buffer) {
            PROTECT(retnames = allocVector(STRSXP, cols_count));
            PROTECT(ret = allocVector(VECSXP, cols_count));
            for (long c = 0; c < cols_count; ++c) {
                SET_STRING_ELT(retnames, c, mkChar(colnames[c].c_str()));
                SET_VECTOR_ELT(ret, c, cols[c]);
            }
            setAttrib(ret, R_NamesSymbol, retnames);
            UNPROTECT(cols_count + 2);
        }
        return ret;
    }

    SEXP RVM_emit_block(SEXP dataexp, SEXP datain) {
        SWIGResultHandler *data = static_cast<SWIGResultHandler*>(R_ExternalPtrAddr(dataexp));
        std::vector<SWIGVM_columntype_t> &coltypes = *(SWIGVM_params->out_types);
        long cols_count = coltypes.size();
        long emiting_rows = 0;
        struct {
            SEXP dat;
            long len;
            void (*fun)(SWIGResultHandler* data, SEXP &inp, long c, long r);
        } setfs[cols_count];

        if (datain == R_NilValue || isNull(datain) || !IS_LIST(datain) || LENGTH(datain) != cols_count) {
            if (datain != R_NilValue && IS_LIST(datain)) {
                stringstream sb;
                sb << "emit function argument count (" << LENGTH(datain)
                   << ") must match the number of output columns (" << cols_count << ')';
                SWIGVM_params->exch->setException(sb.str().c_str());
            } else
                SWIGVM_params->exch->setException("emit function argument count must match the number of output columns");
            return R_NilValue;
        }
        for (long c = 0; c < cols_count; ++c) {
            setfs[c].dat = VECTOR_ELT(datain, c);
            if (isNull(setfs[c].dat)) {
                SWIGVM_params->exch->setException("emitting NULL values not supported");
                return R_NilValue;
            }   
            setfs[c].len = GET_LENGTH(setfs[c].dat);
            if (setfs[c].len < 1) {
                SWIGVM_params->exch->setException("emitting empty vectors not supported");
                return R_NilValue;
            }   
            if (emiting_rows < setfs[c].len)
                emiting_rows = setfs[c].len;
            switch (coltypes[c].type) {
            case INT32     : setfs[c].fun = &RVM_emit_block_set_Int32;     break;
            case INT64     : setfs[c].fun = &RVM_emit_block_set_Int64;     break;
            case DOUBLE    : setfs[c].fun = &RVM_emit_block_set_Double;    break;
            case NUMERIC   : setfs[c].fun = &RVM_emit_block_set_Numeric;   break;
            case DATE      : setfs[c].fun = &RVM_emit_block_set_Date;      break;
            case TIMESTAMP : setfs[c].fun = &RVM_emit_block_set_Timestamp; break;
            case STRING    : setfs[c].fun = &RVM_emit_block_set_String;    break;
            case BOOLEAN   : setfs[c].fun = &RVM_emit_block_set_Boolean;   break;
            default:
                SWIGVM_params->exch->setException("Internal error: wrong column type");
                break;
            }
        }
        for (long r = 0; r < emiting_rows; ++r) {
            for (long c = 0; c < cols_count; ++c)
                setfs[c].fun(data, setfs[c].dat, c, r % setfs[c].len);
            data->next();
        }
        return R_NilValue;
    }
}

RVMImpl::RVMImpl(bool checkOnly): m_checkOnly(checkOnly) {
    char const *argv[] = {"Rcontainer", "--gui=none", "--silent", "--no-save", "--slave"};
    int argc = 5;
    DllInfo *info = NULL;

    setenv("R_HOME", "/usr/lib/R", 1);
    Rf_initEmbeddedR(argc, const_cast<char**>(argv));
    R_Interactive = (Rboolean)0; /* 0 has problems with Exceptions -> needs options(error = ...) ? */
    info = R_getEmbeddingDllInfo();
    R_init_exascript_r(info);

    evaluate_code(integrated_exascript_r_r);
    evaluate_code(integrated_exascript_r_preset_r);
    evaluate_code_protected(SWIGVM_params->script_code);
    evaluate_code(integrated_exascript_r_wrap_r);
}

void RVMImpl::shutdown() {
    SEXP fun, expr;
    int errorOccurred;
    PROTECT(fun = findFun(install("INTERNAL_CLEANUP_WRAPPER__"), R_GlobalEnv));
    PROTECT(expr = allocVector(LANGSXP, 1));
    SETCAR(expr, fun);
    R_tryEvalSilent(expr, R_GlobalEnv, &errorOccurred);
    if (errorOccurred)
        throw RVM::exception(R_curErrorBuf());
    Rf_endEmbeddedR(0);
    R_CleanTempDir();
}

bool RVMImpl::run() {
    SEXP fun, expr;
    int errorOccurred;

    PROTECT(fun = findFun(install("INTERNAL_RUN_WRAPPER__"), R_GlobalEnv));
    PROTECT(expr = allocVector(LANGSXP, 1));
    SETCAR(expr, fun);
    R_tryEvalSilent(expr, R_GlobalEnv, &errorOccurred);
    UNPROTECT(2);
    if (errorOccurred)
        throw RVM::exception(R_curErrorBuf());
    return true;
}

// taken from YAP source code
static SEXP myFindFun(SEXP symb, SEXP envir){
     SEXP fun;
     SEXPTYPE t;
     fun = findVar(symb,envir);
     t = TYPEOF(fun);

     /* eval promise if need be */
     if (t == PROMSXP){
         int error=1;
         fun = R_tryEval(fun,envir,&error);
         if (error) return R_UnboundValue;
         t = TYPEOF(fun);
     }

     if (t == CLOSXP || t == BUILTINSXP || t == BUILTINSXP || t == 
SPECIALSXP)
         return fun;
     return R_UnboundValue;
}


static SEXP import_spec_to_R(ExecutionGraph::ImportSpecification* imp_spec, size_t& num_protects)
{
    // the list element names
    SEXP dimnames = PROTECT(allocVector(VECSXP,6));
    num_protects++;
    SET_VECTOR_ELT(dimnames,0,mkString("is_subselect"));
    SET_VECTOR_ELT(dimnames,1,mkString("subselect_column_names"));
    SET_VECTOR_ELT(dimnames,2,mkString("subselect_column_types"));
    SET_VECTOR_ELT(dimnames,3,mkString("connection_name"));
    SET_VECTOR_ELT(dimnames,4,mkString("connection"));
    SET_VECTOR_ELT(dimnames,5,mkString("parameters"));


    // the list element values ...
    SEXP argsexp = PROTECT(allocVector(VECSXP,6));
    num_protects++;


    // ... isSubselect
    SET_VECTOR_ELT(argsexp,0, ScalarLogical(imp_spec->isSubselect()?TRUE:FALSE));


    if (imp_spec->hasSubselectColumnNames()) {
        // .. subselect_column_names
        size_t numSubselectColumnNames = imp_spec->getSubselectColumnNames().size();
        SEXP sscn = PROTECT(allocVector(VECSXP,numSubselectColumnNames));
        num_protects++;
        for (size_t i=0; i< numSubselectColumnNames; i++)
        {
            SET_VECTOR_ELT(sscn,i,mkString(imp_spec->getSubselectColumnNames()[i].c_str()));
        }
        SET_VECTOR_ELT(argsexp,1,sscn);
    }
    else
    {
        SET_VECTOR_ELT(argsexp,1,R_NilValue);
    }

    if (imp_spec->hasSubselectColumnTypes()) {
        // .. subselect_column_types
        size_t numSubselectColumnTypes = imp_spec->getSubselectColumnTypes().size();
        SEXP ssct = PROTECT(allocVector(VECSXP,numSubselectColumnTypes));
        num_protects++;
        for (size_t i=0; i< numSubselectColumnTypes; i++)
        {
            SET_VECTOR_ELT(ssct,i,mkString(imp_spec->getSubselectColumnTypes()[i].c_str()));
        }
        SET_VECTOR_ELT(argsexp,2,ssct);
    }
    else
    {
        SET_VECTOR_ELT(argsexp,2,R_NilValue);
    }

    if (imp_spec->hasConnectionName())
    {
        // .. connection_name
        SET_VECTOR_ELT(argsexp,3,mkString(imp_spec->getConnectionName().c_str()));
    }
    else
    {
        SET_VECTOR_ELT(argsexp,3,R_NilValue);
    }

    if (imp_spec->hasConnectionInformation())
    {
        // .. connection
        SEXP ci = PROTECT(allocVector(VECSXP,4));
        num_protects++;
        SET_VECTOR_ELT(ci,0,mkString(imp_spec->getConnectionInformation().getKind().c_str()));
        SET_VECTOR_ELT(ci,1,mkString(imp_spec->getConnectionInformation().getAddress().c_str()));
        SET_VECTOR_ELT(ci,2,mkString(imp_spec->getConnectionInformation().getUser().c_str()));
        SET_VECTOR_ELT(ci,3,mkString(imp_spec->getConnectionInformation().getPassword().c_str()));

        SEXP ci_names = PROTECT(allocVector(VECSXP,4));
        num_protects++;
        SET_VECTOR_ELT(ci_names,0,mkString("type"));
        SET_VECTOR_ELT(ci_names,1,mkString("address"));
        SET_VECTOR_ELT(ci_names,2,mkString("user"));
        SET_VECTOR_ELT(ci_names,3,mkString("password"));

        setAttrib(ci, R_NamesSymbol, ci_names);
        SET_VECTOR_ELT(argsexp,4,ci);
    }
    else
    {
        SET_VECTOR_ELT(argsexp,4,R_NilValue);
    }



    // .. parameters
    size_t np = imp_spec->getParameters().size();
    SEXP params = PROTECT(allocVector(VECSXP,np));
    num_protects++;
    SEXP param_names = PROTECT(allocVector(VECSXP,np));
    num_protects++;

    size_t count = 0;
    for (std::map<std::string, std::string>::const_iterator i = imp_spec->getParameters().begin();
         i != imp_spec->getParameters().end();
         ++i)
    {
        SET_VECTOR_ELT(param_names,count,mkString(i->first.c_str()));
        SET_VECTOR_ELT(params,count,mkString(i->second.c_str()));
        count++;
    }
    setAttrib(params, R_NamesSymbol, param_names);
    SET_VECTOR_ELT(argsexp,5,params);



    setAttrib(argsexp, R_NamesSymbol, dimnames);
    return argsexp;
}

static SEXP export_spec_to_R(ExecutionGraph::ExportSpecification* exp_spec, size_t& num_protects)
{
    // the list element names
    const int num_elements = 7;
    SEXP dimnames = PROTECT(allocVector(VECSXP,num_elements));
    num_protects++;
    SET_VECTOR_ELT(dimnames,0,mkString("has_truncate"));
    SET_VECTOR_ELT(dimnames,1,mkString("has_replace"));
    SET_VECTOR_ELT(dimnames,2,mkString("created_by"));
    SET_VECTOR_ELT(dimnames,3,mkString("source_column_names"));
    SET_VECTOR_ELT(dimnames,4,mkString("connection_name"));
    SET_VECTOR_ELT(dimnames,5,mkString("connection"));
    SET_VECTOR_ELT(dimnames,6,mkString("parameters"));

    // the list element values ...
    SEXP argsexp = PROTECT(allocVector(VECSXP,num_elements));
    num_protects++;

    // ... hasTruncate
    SET_VECTOR_ELT(argsexp,0, ScalarLogical(exp_spec->hasTruncate()?TRUE:FALSE));
    // ... hasReplace
    SET_VECTOR_ELT(argsexp,1, ScalarLogical(exp_spec->hasReplace()?TRUE:FALSE));
    // ... createdBy
    if (exp_spec->hasCreatedBy())
    {
        SET_VECTOR_ELT(argsexp,2,mkString(exp_spec->getCreatedBy().c_str()));
    }
    else
    {
        SET_VECTOR_ELT(argsexp,2,R_NilValue);
    }

    // .. source_column_names
    size_t numSourceColumnNames = exp_spec->getSourceColumnNames().size();
    if (numSourceColumnNames > 0) {
        SEXP scn = PROTECT(allocVector(VECSXP,numSourceColumnNames));
        num_protects++;
        for (size_t i=0; i < numSourceColumnNames; i++)
        {
            SET_VECTOR_ELT(scn,i,mkString(exp_spec->getSourceColumnNames()[i].c_str()));
        }
        SET_VECTOR_ELT(argsexp,3,scn);
    }
    else
    {
        SET_VECTOR_ELT(argsexp,3,R_NilValue);
    }

    // .. connection_name
    if (exp_spec->hasConnectionName())
    {
        SET_VECTOR_ELT(argsexp,4,mkString(exp_spec->getConnectionName().c_str()));
    }
    else
    {
        SET_VECTOR_ELT(argsexp,4,R_NilValue);
    }

    // .. connection
    if (exp_spec->hasConnectionInformation())
    {
        SEXP ci = PROTECT(allocVector(VECSXP,4));
        num_protects++;
        SET_VECTOR_ELT(ci,0,mkString(exp_spec->getConnectionInformation().getKind().c_str()));
        SET_VECTOR_ELT(ci,1,mkString(exp_spec->getConnectionInformation().getAddress().c_str()));
        SET_VECTOR_ELT(ci,2,mkString(exp_spec->getConnectionInformation().getUser().c_str()));
        SET_VECTOR_ELT(ci,3,mkString(exp_spec->getConnectionInformation().getPassword().c_str()));

        SEXP ci_names = PROTECT(allocVector(VECSXP,4));
        num_protects++;
        SET_VECTOR_ELT(ci_names,0,mkString("type"));
        SET_VECTOR_ELT(ci_names,1,mkString("address"));
        SET_VECTOR_ELT(ci_names,2,mkString("user"));
        SET_VECTOR_ELT(ci_names,3,mkString("password"));

        setAttrib(ci, R_NamesSymbol, ci_names);
        SET_VECTOR_ELT(argsexp,5,ci);
    }
    else
    {
        SET_VECTOR_ELT(argsexp,5,R_NilValue);
    }

    // .. parameters
    size_t np = exp_spec->getParameters().size();
    SEXP params = PROTECT(allocVector(VECSXP,np));
    num_protects++;
    SEXP param_names = PROTECT(allocVector(VECSXP,np));
    num_protects++;

    size_t count = 0;
    for (std::map<std::string, std::string>::const_iterator i = exp_spec->getParameters().begin();
         i != exp_spec->getParameters().end();
         ++i)
    {
        SET_VECTOR_ELT(param_names,count,mkString(i->first.c_str()));
        SET_VECTOR_ELT(params,count,mkString(i->second.c_str()));
        count++;
    }
    setAttrib(params, R_NamesSymbol, param_names);
    SET_VECTOR_ELT(argsexp,6,params);

    setAttrib(argsexp, R_NamesSymbol, dimnames);
    return argsexp;
}


static string singleCallResult;

const char* RVMImpl::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args, string &calledUndefinedSingleCall) {
    SEXP fun, expr, ret;
    int errorOccurred;

    bool hasArgs = (!args.isEmpty());

    const char* func = NULL;
    switch (fn) {
    case SC_FN_NIL: break;
    case SC_FN_DEFAULT_OUTPUT_COLUMNS: func = "defaultOutputColumns"; break;
    case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL: throw RVM::exception("R is not a supported language for adapter calls"); break;
    case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC: func = "generate_sql_for_import_spec"; break;
    case SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC: func = "generate_sql_for_export_spec"; break;
    }
    if (func == NULL)
    {
        abort();
    }

    size_t num_protects = 0;

    PROTECT(expr = allocVector(LANGSXP, hasArgs?3:2));
    num_protects++;

    SEXP userFun;
    userFun = myFindFun(install(func), R_GlobalEnv);
    if (userFun == R_UnboundValue) {
        UNPROTECT(num_protects);
        calledUndefinedSingleCall = func;
        return strdup("<error>");

        //throw swig_undefined_single_call_exception(func); 
    }

    PROTECT(fun = myFindFun(install("INTERNAL_SINGLE_CALL_WRAPPER__"), R_GlobalEnv));
    num_protects++;
    if (fun == R_UnboundValue) {
        abort(); 
    }
    SETCAR(expr, fun);

    SETCADR(expr, mkString(func));

    if (hasArgs)
    {
        ExecutionGraph::ImportSpecification* imp_spec =
                const_cast<ExecutionGraph::ImportSpecification*>(
                    dynamic_cast<const ExecutionGraph::ImportSpecification*>(&args));

        ExecutionGraph::ExportSpecification* exp_spec =
                const_cast<ExecutionGraph::ExportSpecification*>(
                    dynamic_cast<const ExecutionGraph::ExportSpecification*>(&args));

         if (imp_spec)
         {
             SETCADDR(expr,import_spec_to_R(imp_spec,num_protects));
         }
         else if (exp_spec)
         {
             SETCADDR(expr,export_spec_to_R(exp_spec,num_protects));
         }
         else
         {
             throw RVM::exception("Internal R VM error: cannot cast argument DTO to import/export specification");
         }
    }
    PROTECT(ret = R_tryEvalSilent(expr, R_GlobalEnv, &errorOccurred));
    num_protects++;

    if (errorOccurred)
    {
         UNPROTECT(num_protects);
        throw RVM::exception(R_curErrorBuf());
    }

    if (!isString(ret)) {
        UNPROTECT(num_protects);
        throw RVM::exception("result of singleCall function is not of type string");
    }

    SEXP c_res = STRING_ELT(ret,0);
    singleCallResult = string(CHAR(c_res));

    UNPROTECT(num_protects);
    return singleCallResult.c_str();
}

