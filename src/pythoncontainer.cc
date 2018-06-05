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

extern "C" void init_exascript_python(void);

static void check() {
    PyObject *pt, *pv, *tb, *s = NULL, *pvc, *pvcn;
    string pvcns("");
    if (PyErr_Occurred() == NULL) return;
    PyErr_Fetch(&pt, &pv, &tb); if (pt == NULL) return;
    PyErr_NormalizeException(&pt, &pv, &tb); if (pt == NULL) return;
    s = PyObject_Str(pv);
    if (NULL != (pvc = PyObject_GetAttrString(pv, "__class__"))) {
        if (NULL != (pvcn = PyObject_GetAttrString(pvc, "__name__"))) {
            pvcns = string(PyString_AS_STRING(pvcn)) + string(": ");
            Py_XDECREF(pvcn);
        }
        Py_XDECREF(pvc);
    }
    PythonVM::exception x((pvcns + PyString_AS_STRING(s)).c_str());
    Py_XDECREF(s);
    PyErr_Clear();
    throw x;
}

class SWIGVMContainers::PythonVMImpl {
    public:
        PythonVMImpl(bool checkOnly);
        ~PythonVMImpl() {}
        bool run();
        std::string singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args);
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

void PythonVM::shutdown() {m_impl->shutdown();}

bool PythonVM::run() {
    try {
        return m_impl->run();
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }
    return false;
}
std::string PythonVM::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) {
    try {
        return m_impl->singleCall(fn, args);
    } catch (std::exception& err) {
        lock_guard<mutex> lock(exception_msg_mtx);
        exception_msg = err.what();
    }
    return "<this is an error>";
}


#ifndef DISABLE_PYTHON_SUBINTERP
PyThreadState *PythonVMImpl::main_thread = NULL;
#endif

PythonVMImpl::PythonVMImpl(bool checkOnly): m_checkOnly(checkOnly)
{
//    script_code = string("\xEF\xBB\xBF") + string(SWIGVM_params->script_code);
    script_code = string("\xEF\xBB\xBF") + string(SWIGVM_params->script_code);

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
        Py_Initialize();
        PyEval_InitThreads();
#ifndef DISABLE_PYTHON_SUBINTERP
        main_thread = PyEval_SaveThread();
#endif
    }

    globals = PyDict_New();

//    PyObject *main_module = PyImport_AddModule("__main__");
//    if (main_module == nullptr) {
//        throw PythonVM::exception("Failed to get Python main module");
//    }
    //globals = PyModule_GetDict(main_module);

    //PyRun_String("import sys, types, os", Py_single_input,globals, globals);
    //PyRun_String("sys.modules.setdefault('google', types.ModuleType('google'))", Py_single_input, globals, globals);
    //PyRun_String("site.main()", Py_single_input, globals, globals);

    //PyRun_String("import sys, types, os;has_mfs = sys.version_info > (3, 5);p = os.path.join(sys._getframe(1).f_locals['sitedir'], *('google',));importlib = has_mfs and __import__('importlib.util');has_mfs and __import__('importlib.machinery');m = has_mfs and sys.modules.setdefault('google', importlib.util.module_from_spec(importlib.machinery.PathFinder.find_spec('google', [os.path.dirname(p)])));m = m or sys.modules.setdefault('google', types.ModuleType('google'));mp = (m or []) and m.__dict__.setdefault('__path__',[]);(p not in mp) and mp.append(p)", Py_single_input,globals, globals);

    {   
#ifndef DISABLE_PYTHON_SUBINTERP
        PythonThreadBlock block;
#endif
        
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

        PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins());
        init_exascript_python();

        code = Py_CompileString(integrated_exascript_python_py, "exascript_python.py", Py_file_input); check();
        if (code == NULL) throw PythonVM::exception("Failed to compile internal module");
        exatable = PyImport_ExecCodeModule((char*)"exascript_python", code); check();

        code = Py_CompileString(integrated_exascript_python_preset_py, "<EXASCRIPTPP>", Py_file_input); check();
        if (code == NULL) throw PythonVM::exception("Failed to compile preset script");
        PyEval_EvalCode(reinterpret_cast<PyCodeObject*>(code), globals, globals); check();
        Py_DECREF(code);

        //PyEval_EvalCode(reinterpret_cast<PyCodeObject*>(script), globals, globals); check();
        PyObject *runobj = PyDict_GetItemString(globals, "__pythonvm_wrapped_parse"); check();
        PyObject *retvalue = PyObject_CallFunction(runobj, NULL); check();
        Py_XDECREF(retvalue); retvalue = NULL;

        code = Py_CompileString(integrated_exascript_python_wrap_py, "<EXASCRIPT>", Py_file_input); check();
        if (code == NULL) throw PythonVM::exception("Failed to compile wrapping script");
        PyEval_EvalCode(reinterpret_cast<PyCodeObject*>(code), globals, globals); check();
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
        Py_XDECREF(retvalue); retvalue = NULL;
    }
    return true;
}

std::string PythonVMImpl::singleCall(single_call_function_id_e fn, const ExecutionGraph::ScriptDTO& args) {
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
            throw swig_undefined_single_call_exception(func);  // no such call is defined.
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
        const char * s = PyString_AsString(retvalue);
        std::string result(s);
        Py_XDECREF(retvalue); retvalue = NULL;
        return result;
}
