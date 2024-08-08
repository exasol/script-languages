#include "pythoncontainer.h"
#include "exaudf/exaudflib/swig/swig_meta_data.h"
#include <iostream>
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#include <Python.h>
#include "exascript_python_int.h"
#include "exascript_python.h"
#include "exaudf/debug_message.h"
#include "exaudf/exaudflib/vm/scriptoptionlines.h"

#include "exaudf/exaudflib/swig/script_data_transfer_objects.h"

#include <sstream>
#include <string.h>

#define DISABLE_PYTHON_SUBINTERP

using namespace SWIGVMContainers;
using namespace std;

extern "C" PyObject* PyInit__exascript_python(void);


static void check(const std::string& error_code) {
    PyObject *pt, *pv, *tb, *s = NULL, *pvc, *pvcn;
    string pvcns("");
    if (PyErr_Occurred() == NULL) return;
    PyErr_Fetch(&pt, &pv, &tb); if (pt == NULL) return;
    PyErr_NormalizeException(&pt, &pv, &tb); if (pt == NULL) return;
    s = PyObject_Str(pv);
    
    // Get Exception name
    if (NULL != (pvc = PyObject_GetAttrString(pv, "__class__"))) {
        if (NULL != (pvcn = PyObject_GetAttrString(pvc, "__name__"))) {
	  PyObject* repr = PyObject_Str(pvcn);
	  PyObject* p3str = PyUnicode_AsEncodedString(repr, "utf-8", "ignore");
	  const char *bytes = PyBytes_AS_STRING(p3str);
	  pvcns = string(bytes) + string(": ");
	  Py_XDECREF(pvcn);
        }
        Py_XDECREF(pvc);
    }


    string exception_string("");
    PyObject* repr = PyObject_Str(s);
    PyObject* p3str = PyUnicode_AsEncodedString(repr, "utf-8", "ignore");
    const char *bytes = PyBytes_AS_STRING(p3str);
    exception_string = error_code+": "+pvcns + string(bytes);
    PythonVM::exception x(exception_string.c_str());
    Py_XDECREF(s);
    PyErr_Clear();
    throw x;
}

class SWIGVMContainers::PythonVMImpl {
    public:
        PythonVMImpl(bool checkOnly);
        ~PythonVMImpl() {}
        bool run();
        const char* singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args, string& calledUndefinedSingleCall);
        void shutdown();
    private:
        string script_code;
        bool m_checkOnly;
        PyObject *globals, *code, *script;
        PyObject *exatable, *runobj, *cleanobj, *clean_wrap_obj;
        PyObject *retvalue;
#ifndef DISABLE_PYTHON_SUBINTERP
        PyThreadState *pythread;
        static PyThreadState *main_thread;
#endif
};

#ifndef DISABLE_PYTHON_SUBINTERP
class PythonThreadBlock {
    PyGILState_STATE state;
    PyThreadState *save;
    public:
        PythonThreadBlock(): state(PyGILState_Ensure()), save(PyThreadState_Get()) {}
        ~PythonThreadBlock() { PyThreadState_Swap(save); PyGILState_Release(state); }
};
#endif

PythonVM::PythonVM(bool checkOnly) {
    try {
        DBG_FUNC_CALL(cerr, m_impl = new PythonVMImpl(checkOnly));
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        DBG_EXCEPTION(cerr, err);
        exception_msg = "F-UDF-CL-SL-PYTHON-1000: " + std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-PYTHON-1125: python crashed for unknown reasons";
    }

}

void PythonVM::shutdown() {
    try {
       DBG_FUNC_CALL(cerr, m_impl->shutdown());
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-PYTHON-1001: " + std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-PYTHON-1124: python crashed for unknown reasons";
    }
}

bool PythonVM::run() {
    try {
        DBG_FUNC_CALL(cerr, bool result = m_impl->run());
        return result; 
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-PYTHON-1002: "+ std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-PYTHON-1003: python crashed for unknown reasons";
    }
    return false;
}


const char* PythonVM::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) {
    try {
        return m_impl->singleCall(fn, args,calledUndefinedSingleCall);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-PYTHON-1004: "+std::string(err.what());
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "F-UDF-CL-SL-PYTHON-1123: python crashed for unknown reasons";
    }
    return strdup("<this is an error>");
}


#ifndef DISABLE_PYTHON_SUBINTERP
PyThreadState *PythonVMImpl::main_thread = NULL;
#endif

PythonVMImpl::PythonVMImpl(bool checkOnly): m_checkOnly(checkOnly)
{
    DBG_FUNC_BEGIN( cerr );
    script_code = string("\xEF\xBB\xBF") + string(SWIGVM_params->script_code);    // Magic-Number of UTF-8 files

    script = exatable = globals = retvalue = NULL;
#ifndef DISABLE_PYTHON_SUBINTERP
    pythread = NULL;
#endif

    if (!Py_IsInitialized()) {
        ::setlocale(LC_ALL, "en_US.utf8");
        Py_NoSiteFlag = 1;
        PyImport_AppendInittab("_exascript_python",PyInit__exascript_python);
        Py_Initialize();
        PyEval_InitThreads();
#ifndef DISABLE_PYTHON_SUBINTERP
        main_thread = PyEval_SaveThread();
#endif
    }

    
    {   
#ifndef DISABLE_PYTHON_SUBINTERP
        PythonThreadBlock block;
#endif
        
	globals = PyDict_New();
	PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());
	script = Py_CompileString(script_code.c_str(), SWIGVM_params->script_name, Py_file_input); check("F-UDF-CL-SL-PYTHON-1005");
        if (script == NULL) throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1006: Failed to compile script");

#ifndef DISABLE_PYTHON_SUBINTERP
        pythread = PyThreadState_New(main_thread->interp);
        if (pythread == NULL)
            throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1007: Failed to create Python interpreter");
#endif
    }

    if (!checkOnly) {
#ifndef DISABLE_PYTHON_SUBINTERP
        PythonThreadBlock block;
        PyThreadState_Swap(pythread);
#endif
        code = Py_CompileString(integrated_exascript_python_py, "exascript_python.py", Py_file_input); check("F-UDF-CL-SL-PYTHON-1008");
        if (code == NULL) throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1009: Failed to compile internal module");
        exatable = PyImport_ExecCodeModule((char*)"exascript_python", code);
	check("F-UDF-CL-SL-PYTHON-1010");
	if (exatable == NULL) throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1011: Failed to import code module");

        code = Py_CompileString(integrated_exascript_python_preset_py, "<EXASCRIPTPP>", Py_file_input); check("F-UDF-CL-SL-PYTHON-1012");
        if (code == NULL) {check("F-UDF-CL-SL-PYTHON-1013");}

 	PyEval_EvalCode(code, globals, globals); check("F-UDF-CL-SL-PYTHON-1014"); 
        Py_DECREF(code);

         PyObject *runobj = PyDict_GetItemString(globals, "__pythonvm_wrapped_parse"); check("F-UDF-CL-SL-PYTHON-1016");
         //PyObject *retvalue = PyObject_CallFunction(runobj, NULL); check();
	 PyObject *retvalue = PyObject_CallFunctionObjArgs(runobj, globals, NULL); check("F-UDF-CL-SL-PYTHON-1017");
         Py_XDECREF(retvalue); retvalue = NULL;

	code = Py_CompileString(integrated_exascript_python_wrap_py, "<EXASCRIPT>", Py_file_input); check("F-UDF-CL-SL-PYTHON-1018");
        if (code == NULL) throw PythonVM::exception("Failed to compile wrapping script");

	PyEval_EvalCode(code, globals, globals); check("F-UDF-CL-SL-PYTHON-1019");
        Py_XDECREF(code); 
    }
    DBG_FUNC_END( cerr );
}

void PythonVMImpl::shutdown() {
    {   
#ifndef DISABLE_PYTHON_SUBINTERP
        PythonThreadBlock block;
        if (pythread != NULL)
            PyThreadState_Swap(pythread);
#endif
        Py_XDECREF(retvalue);
        if (!m_checkOnly) {
            cleanobj = PyDict_GetItemString(globals, "cleanup"); check("F-UDF-CL-SL-PYTHON-1021");
            if (cleanobj){
                clean_wrap_obj = PyDict_GetItemString(globals, "__pythonvm_wrapped_cleanup"); check("F-UDF-CL-SL-PYTHON-1022");
                if (clean_wrap_obj) {
                    retvalue = PyObject_CallObject(clean_wrap_obj, NULL);
                    check("F-UDF-CL-SL-PYTHON-1023");
                } 
            }
        }
        Py_XDECREF(retvalue); retvalue = NULL;
        Py_XDECREF(script);
        Py_XDECREF(exatable);
        Py_XDECREF(globals);
    }
}

bool PythonVMImpl::run() {
    DBG_FUNC_BEGIN( cerr );

    if (m_checkOnly) throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1024: Python VM in check only mode");

    {
#ifndef DISABLE_PYTHON_SUBINTERP
        PythonThreadBlock block;
        PyThreadState_Swap(pythread);
#endif
        DBG_FUNC_CALL(cerr, runobj = PyDict_GetItemString(globals, "__pythonvm_wrapped_run")); check("F-UDF-CL-SL-PYTHON-1025");
        DBG_FUNC_CALL(cerr, retvalue = PyObject_CallFunction(runobj, NULL)); check("F-UDF-CL-SL-PYTHON-1026");
	if (retvalue == NULL) {
	  throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1027: Python VM: calling 'run' failed without an exception)");
	}
        Py_XDECREF(retvalue); retvalue = NULL;
    }
    return true;
}


static string singleCallResult;

const char* PythonVMImpl::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args , string& calledUndefinedSingleCall) {
    if (m_checkOnly) throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1028: Python VM in check only mode (singleCall)"); // @@@@ TODO: better exception text
    //{
#ifndef DISABLE_PYTHON_SUBINTERP
        PythonThreadBlock block;
        PyThreadState_Swap(pythread);
#endif

        const char* func = NULL;
        switch (fn) {
        case SC_FN_NIL: break;
        case SC_FN_DEFAULT_OUTPUT_COLUMNS: func = "default_output_columns"; break;
        case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL: func = "adapter_call"; break;
        case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC: func = "generate_sql_for_import_spec"; break;
        case SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC: func = "generate_sql_for_export_spec"; break;
        }
        if (func == NULL)
        {
            throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1029: Unknown single call function "+fn);
        }
        PyObject* argObject = NULL;

        if (fn==SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC)
        {
            argObject= PyDict_New();
            const ExecutionGraph::ImportSpecification* imp_spec = dynamic_cast<const ExecutionGraph::ImportSpecification*>(&args);
            if (imp_spec == NULL)
            {
                throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1030: Internal Python VM error: cannot cast argument DTO to import specification");
            }
            //        import_spec.is_subselect
            PyDict_SetItemString(argObject,"is_subselect", (imp_spec->isSubselect())?Py_True:Py_False);

             if (imp_spec->hasSubselectColumnNames()) {
                 size_t numSubselectColumnNames = imp_spec->getSubselectColumnNames().size();
                 PyObject* names = PyList_New(numSubselectColumnNames);
                 for (size_t i=0; i< numSubselectColumnNames; i++)
                 {
                     PyList_SetItem(names,i,PyString_FromString(imp_spec->getSubselectColumnNames()[i].c_str()));
                 }
                 PyDict_SetItemString(argObject,"subselect_column_names",names);
                 Py_XDECREF(names);
             } else {
                 PyDict_SetItemString(argObject,"subselect_column_names", Py_None);
             }
             if (imp_spec->hasSubselectColumnTypes()) {
                 size_t numSubselectColumnTypes = imp_spec->getSubselectColumnTypes().size();
                 PyObject* types = PyList_New(numSubselectColumnTypes);
                 for (size_t i=0; i< numSubselectColumnTypes; i++)
                 {
                     PyList_SetItem(types,i,PyString_FromString(imp_spec->getSubselectColumnTypes()[i].c_str()));
                 }
                 PyDict_SetItemString(argObject,"subselect_column_types",types);
                 Py_XDECREF(types);
             } else {
                 PyDict_SetItemString(argObject,"subselect_column_types", Py_None);
             }

             if (imp_spec->hasConnectionName()) {
                 PyObject* connection_name = PyString_FromString(imp_spec->getConnectionName().c_str());
                 PyDict_SetItemString(argObject,"connection_name",connection_name);
                 Py_XDECREF(connection_name);
             } else {
                 PyDict_SetItemString(argObject,"connection_name", Py_None);
             }

             if (imp_spec->hasConnectionInformation()) {
                 PyObject* connectionObject = PyDict_New();

                 PyObject* kind = PyString_FromString(imp_spec->getConnectionInformation().getKind().c_str());
                 PyDict_SetItemString(connectionObject,"type",kind);
                 Py_XDECREF(kind);

                 PyObject* address = PyString_FromString(imp_spec->getConnectionInformation().getAddress().c_str());
                 PyDict_SetItemString(connectionObject,"address",address);
                 Py_XDECREF(address);

                 PyObject* user = PyString_FromString(imp_spec->getConnectionInformation().getUser().c_str());
                 PyDict_SetItemString(connectionObject,"user",user);
                 Py_XDECREF(user);

                 PyObject* password = PyString_FromString(imp_spec->getConnectionInformation().getPassword().c_str());
                 PyDict_SetItemString(connectionObject,"password",password);
                 Py_XDECREF(password);

                 PyDict_SetItemString(argObject,"connection",connectionObject);
                 Py_XDECREF(connectionObject);
             } else {
                 PyDict_SetItemString(argObject,"connection", Py_None);
             }

             PyObject* paramObject= PyDict_New();
             for (std::map<std::string, std::string>::const_iterator i = imp_spec->getParameters().begin();
                  i != imp_spec->getParameters().end();
                  ++i)
             {
                 PyObject* key = PyString_FromString(i->first.c_str());
                 PyObject* value = PyString_FromString(i->second.c_str());
                 PyDict_SetItem(paramObject,key,value);
                 Py_XDECREF(key);
                 Py_XDECREF(value);
             }
             PyDict_SetItemString(argObject,"parameters",paramObject);
             Py_XDECREF(paramObject);
        } else if (fn==SC_FN_GENERATE_SQL_FOR_EXPORT_SPEC) {
            argObject= PyDict_New();
            const ExecutionGraph::ExportSpecification* exp_spec = dynamic_cast<const ExecutionGraph::ExportSpecification*>(&args);
            if (exp_spec == NULL)
            {
                throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1031: Internal Python VM error: cannot cast argument DTO to export specification");
            }

            if (exp_spec->hasConnectionName()) {
                PyObject* connection_name = PyString_FromString(exp_spec->getConnectionName().c_str());
                PyDict_SetItemString(argObject,"connection_name",connection_name);
                Py_XDECREF(connection_name);
            } else {
                PyDict_SetItemString(argObject,"connection_name", Py_None);
            }

            if (exp_spec->hasConnectionInformation()) {
                PyObject* connectionObject = PyDict_New();

                PyObject* kind = PyString_FromString(exp_spec->getConnectionInformation().getKind().c_str());
                PyDict_SetItemString(connectionObject,"type",kind);
                Py_XDECREF(kind);

                PyObject* address = PyString_FromString(exp_spec->getConnectionInformation().getAddress().c_str());
                PyDict_SetItemString(connectionObject,"address",address);
                Py_XDECREF(address);

                PyObject* user = PyString_FromString(exp_spec->getConnectionInformation().getUser().c_str());
                PyDict_SetItemString(connectionObject,"user",user);
                Py_XDECREF(user);

                PyObject* password = PyString_FromString(exp_spec->getConnectionInformation().getPassword().c_str());
                PyDict_SetItemString(connectionObject,"password",password);
                Py_XDECREF(password);

                PyDict_SetItemString(argObject,"connection",connectionObject);
                Py_XDECREF(connectionObject);
            } else {
                PyDict_SetItemString(argObject,"connection", Py_None);
            }

            PyObject* paramObject= PyDict_New();
            for (std::map<std::string, std::string>::const_iterator i = exp_spec->getParameters().begin();
                 i != exp_spec->getParameters().end();
                 ++i)
            {
                PyObject* key = PyString_FromString(i->first.c_str());
                PyObject* value = PyString_FromString(i->second.c_str());
                PyDict_SetItem(paramObject,key,value);
                Py_XDECREF(key);
                Py_XDECREF(value);
            }
            PyDict_SetItemString(argObject,"parameters",paramObject);
            Py_XDECREF(paramObject);

            PyDict_SetItemString(argObject,"has_truncate", (exp_spec->hasTruncate())?Py_True:Py_False);
            PyDict_SetItemString(argObject,"has_replace", (exp_spec->hasReplace())?Py_True:Py_False);
            if (exp_spec->hasCreatedBy()) {
                PyObject* created_by = PyString_FromString(exp_spec->getCreatedBy().c_str());
                PyDict_SetItemString(argObject,"created_by",created_by);
                Py_XDECREF(created_by);
            } else {
                PyDict_SetItemString(argObject,"created_by", Py_None);
            }

            size_t numSourceColumnNames = exp_spec->getSourceColumnNames().size();
            if (numSourceColumnNames > 0) {
                PyObject* names = PyList_New(numSourceColumnNames);
                for (size_t i=0; i < numSourceColumnNames; i++)
                {
                    PyList_SetItem(names,i,PyString_FromString(exp_spec->getSourceColumnNames()[i].c_str()));
                }
                PyDict_SetItemString(argObject,"source_column_names",names);
                Py_XDECREF(names);
            } else {
                PyDict_SetItemString(argObject,"source_column_names", Py_None);
            }
        }

//        Py_XINCREF(argObject);

        PyObject* funcToCall = PyDict_GetItemString(globals, func); check("F-UDF-CL-SL-PYTHON-1032");
        if (funcToCall == NULL) {
            calledUndefinedSingleCall = func;
            return strdup("<error>");
            //throw swig_undefined_single_call_exception(func);  // no such call is defined.
        }
        if (fn==SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL) {
            // Call directly
            // TODO VS This will all be refactored
            const ExecutionGraph::StringDTO* argDto = dynamic_cast<const ExecutionGraph::StringDTO*>(&args);
            string string_arg = argDto->getArg();
            runobj = PyDict_GetItemString(globals, func); check("F-UDF-CL-SL-PYTHON-1033");
            retvalue = PyObject_CallFunction(runobj, (char *)"s", string_arg.c_str());
        } else {
            runobj = PyDict_GetItemString(globals, "__pythonvm_wrapped_singleCall"); check("F-UDF-CL-SL-PYTHON-1034");
            if (runobj == NULL) {
                throw PythonVM::exception("F-UDF-CL-SL-PYTHON-1035: Cannot find function __pythonvm_wrapped_singleCall");
            }
            // Call indirectly
            if (argObject == NULL) {
                retvalue = PyObject_CallFunctionObjArgs(runobj, funcToCall, Py_None, NULL);
            } else {
                retvalue = PyObject_CallFunctionObjArgs(runobj, funcToCall, argObject, NULL);
            }
        }
        check("F-UDF-CL-SL-PYTHON-1036");

        Py_XDECREF(argObject);

        if (!PyString_Check(retvalue) && !PyUnicode_Check(retvalue))
        {
            std::stringstream sb;
            sb << "F-UDF-CL-SL-PYTHON-1037: ";
            sb << fn;
            sb << " did not return string type (singleCall)";
            throw PythonVM::exception(sb.str().c_str());
        }
	
        PyObject* repr = PyObject_Str(retvalue);
        PyObject* p3str = PyUnicode_AsEncodedString(repr, "utf-8", "ignore");
        const char *bytes = PyBytes_AS_STRING(p3str);
        singleCallResult = string(bytes);
        Py_XDECREF(retvalue); retvalue = NULL;
	return singleCallResult.c_str();
}
