#include <exaudflib.h>
#include <iostream>
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#include <Python.h>
#include <exascript_python.h>
#include <exascript_python_int.h>
#include "scriptoptionlines.h"

#include "script_data_transfer_objects.h"

#define DISABLE_PYTHON_SUBINTERP

using namespace SWIGVMContainers;
using namespace std;

#ifdef ENABLE_PYTHON3
extern "C" PyObject* PyInit__exascript_python(void);
#else
extern "C" void init_exascript_python(void);
#endif


static void check() {
    PyObject *pt, *pv, *tb, *s = NULL, *pvc, *pvcn;
    string pvcns("");
    if (PyErr_Occurred() == NULL) return;
    PyErr_Fetch(&pt, &pv, &tb); if (pt == NULL) return;
    PyErr_NormalizeException(&pt, &pv, &tb); if (pt == NULL) return;
    s = PyObject_Str(pv);
    if (NULL != (pvc = PyObject_GetAttrString(pv, "__class__"))) {
        if (NULL != (pvcn = PyObject_GetAttrString(pvc, "__name__"))) {
#ifdef ENABLE_PYTHON3
	  PyObject* repr = PyObject_Str(pvcn);
	  PyObject* p3str = PyUnicode_AsEncodedString(repr, "utf-8", "ignore");
	  const char *bytes = PyBytes_AS_STRING(p3str);
	  pvcns = string(bytes) + string(": ");
	  
#else
	  pvcns = string(PyString_AS_STRING(pvcn)) + string(": ");
#endif
	  Py_XDECREF(pvcn);
        }
        Py_XDECREF(pvc);
    }
    string exception_string("");
#ifdef ENABLE_PYTHON3
    PyObject* repr = PyObject_Str(s);
    PyObject* p3str = PyUnicode_AsEncodedString(repr, "utf-8", "ignore");
    const char *bytes = PyBytes_AS_STRING(p3str);
    exception_string = pvcns + string(bytes);
#else
    exception_string = pvcns + PyString_AS_STRING(s);
#endif
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
        PyObject *exatable, *runobj, *cleanobj;
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
        m_impl = new PythonVMImpl(checkOnly);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }

}

void PythonVM::shutdown() {
    try {
        m_impl->shutdown();
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }
}

bool PythonVM::run() {
    try {
        return m_impl->run();
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    } catch (...) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = "python crashed for unknown reasons";
    }
    return false;
}


const char* PythonVM::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) {
    try {
        return m_impl->singleCall(fn, args,calledUndefinedSingleCall);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }
    return strdup("<this is an error>");
}


#ifndef DISABLE_PYTHON_SUBINTERP
PyThreadState *PythonVMImpl::main_thread = NULL;
#endif

PythonVMImpl::PythonVMImpl(bool checkOnly): m_checkOnly(checkOnly)
{
    script_code = string("\xEF\xBB\xBF") + string(SWIGVM_params->script_code);    // Magic-Number of UTF-8 files

//    const string nositeKeyword = "%nosite";
//    const string whitespace = " \t\f\v";
//    const string lineEnd = ";";
//    size_t pos;
//    string nosite = ExecutionGraph::extractOptionLine(script_code, nositeKeyword, whitespace, lineEnd, pos, [&](const char* msg){throw PythonVM::exception(msg);});

//    cerr << "VALUE of nosite: |" << nosite << "|" << endl;


    int noSiteFlag = 0;
//    if (nosite == "yes") {noSiteFlag=1;}
//    else if (nosite == "" || nosite == "no") {noSiteFlag=0;}
//    else throw PythonVM::exception("Invalid value for %nosite option, must be yes or no");

//    cerr << "Value of noSiteFlag: |" << noSiteFlag << "|" << endl;

//    script_code = string("\xEF\xBB\xBF") + script_code;

//    cerr << "Script code after extract option line: " << endl << script_code << endl;


    script = exatable = globals = retvalue = NULL;
#ifndef DISABLE_PYTHON_SUBINTERP
    pythread = NULL;
#endif

    if (!Py_IsInitialized()) {
        ::setlocale(LC_ALL, "en_US.utf8");
        Py_NoSiteFlag = noSiteFlag;
#ifdef ENABLE_PYTHON3
        PyImport_AppendInittab("_exascript_python",PyInit__exascript_python);
#endif
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
	script = Py_CompileString(script_code.c_str(), SWIGVM_params->script_name, Py_file_input); check();
        if (script == NULL) throw PythonVM::exception("Failed to compile script");

#ifndef DISABLE_PYTHON_SUBINTERP
        pythread = PyThreadState_New(main_thread->interp);
        if (pythread == NULL)
            throw PythonVM::exception("Failed to create Python interpreter");
#endif
    }

    if (!checkOnly) {
#ifndef DISABLE_PYTHON_SUBINTERP
        PythonThreadBlock block;
        PyThreadState_Swap(pythread);
#endif


#ifndef ENABLE_PYTHON3
         init_exascript_python();
#endif
        code = Py_CompileString(integrated_exascript_python_py, "exascript_python.py", Py_file_input); check();
        if (code == NULL) throw PythonVM::exception("Failed to compile internal module");
        exatable = PyImport_ExecCodeModule((char*)"exascript_python", code);
	check();
	if (exatable == NULL) throw PythonVM::exception("Failed to import code module");

        code = Py_CompileString(integrated_exascript_python_preset_py, "<EXASCRIPTPP>", Py_file_input); check();
        if (code == NULL) {check();}

 #ifdef ENABLE_PYTHON3
 	PyEval_EvalCode(code, globals, globals); check();
 #else
	PyEval_EvalCode(reinterpret_cast<PyCodeObject*>(code), globals, globals); check();
 #endif
        Py_DECREF(code);

         PyObject *runobj = PyDict_GetItemString(globals, "__pythonvm_wrapped_parse"); check();
         //PyObject *retvalue = PyObject_CallFunction(runobj, NULL); check();
	 PyObject *retvalue = PyObject_CallFunctionObjArgs(runobj, globals, NULL); check();
         Py_XDECREF(retvalue); retvalue = NULL;

	code = Py_CompileString(integrated_exascript_python_wrap_py, "<EXASCRIPT>", Py_file_input); check();
        if (code == NULL) throw PythonVM::exception("Failed to compile wrapping script");

#ifdef ENABLE_PYTHON3
	PyEval_EvalCode(code, globals, globals); check();
#else
        PyEval_EvalCode(reinterpret_cast<PyCodeObject*>(code), globals, globals); check();
#endif

        Py_XDECREF(code); 
    }
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
            cleanobj = PyDict_GetItemString(globals, "cleanup");
            if (cleanobj) {
                retvalue = PyObject_CallObject(cleanobj, NULL);
                check();
            }  
        }
        Py_XDECREF(script);
        Py_XDECREF(exatable);
        Py_XDECREF(globals);
    }
}

bool PythonVMImpl::run() {


    if (m_checkOnly) throw PythonVM::exception("Python VM in check only mode");

    {
#ifndef DISABLE_PYTHON_SUBINTERP
        PythonThreadBlock block;
        PyThreadState_Swap(pythread);
#endif
        runobj = PyDict_GetItemString(globals, "__pythonvm_wrapped_run"); check();
        retvalue = PyObject_CallFunction(runobj, NULL); check();
	if (retvalue == NULL) {
	  throw PythonVM::exception("Python VM: calling 'run' failed without an exception)");
	}
        Py_XDECREF(retvalue); retvalue = NULL;
    }
    return true;
}


static string singleCallResult;

const char* PythonVMImpl::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args , string& calledUndefinedSingleCall) {
    if (m_checkOnly) throw PythonVM::exception("Python VM in check only mode (singleCall)"); // @@@@ TODO: better exception text
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
            abort();
        }
        PyObject* argObject = NULL;

        if (fn==SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC)
        {
            argObject= PyDict_New();
            const ExecutionGraph::ImportSpecification* imp_spec = dynamic_cast<const ExecutionGraph::ImportSpecification*>(&args);
            if (imp_spec == NULL)
            {
                throw PythonVM::exception("Internal Python VM error: cannot cast argument DTO to import specification");
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
                throw PythonVM::exception("Internal Python VM error: cannot cast argument DTO to export specification");
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

        PyObject* funcToCall = PyDict_GetItemString(globals, func); check();
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
            runobj = PyDict_GetItemString(globals, func); check();
            retvalue = PyObject_CallFunction(runobj, (char *)"s", string_arg.c_str());
        } else {
            runobj = PyDict_GetItemString(globals, "__pythonvm_wrapped_singleCall"); check();
            if (runobj == NULL) {
                abort();
            }
            // Call indirectly
            if (argObject == NULL) {
                retvalue = PyObject_CallFunctionObjArgs(runobj, funcToCall, Py_None, NULL);
            } else {
                retvalue = PyObject_CallFunctionObjArgs(runobj, funcToCall, argObject, NULL);
            }
        }
        check();

        Py_XDECREF(argObject);

        if (!PyString_Check(retvalue) && !PyUnicode_Check(retvalue))
        {
            std::stringstream sb;
            sb << fn;
            sb << " did not return string type (singleCall)";
            throw PythonVM::exception(sb.str().c_str());
        }
	
#ifdef ENABLE_PYTHON3
	  PyObject* repr = PyObject_Str(retvalue);
	  PyObject* p3str = PyUnicode_AsEncodedString(repr, "utf-8", "ignore");
	  const char *bytes = PyBytes_AS_STRING(p3str);
	  singleCallResult = string(bytes);
#else
        const char * s = PyString_AsString(retvalue);
        singleCallResult = string(s);
#endif
        Py_XDECREF(retvalue); retvalue = NULL;
	return singleCallResult.c_str();
}
