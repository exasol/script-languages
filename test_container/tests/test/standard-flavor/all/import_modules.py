#!/usr/bin/env python3
from typing import List

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf.udf_debug import UdfDebugger

class ImportAllModulesTest(udf.TestCase):

    def setUp(self):
        self.query('create schema import_all_modules', ignore_errors=True)

    def get_all_root_modules(self) -> List[str]:
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT import_all_modules.get_all_root_modules() 
            EMITS (module_name VARCHAR(200000)) AS

            import sys
            import pkgutil

            def get_module_names(path):
                modules = list(module for _, module, _ in pkgutil.iter_modules(path))
                return modules

            def run(ctx):
                modules = get_module_names(sys.path)
                for module in modules:
                    ctx.emit(module)
            /
            '''))
        rows = self.query('''SELECT import_all_modules.get_all_root_modules() FROM dual''')
        print("Number of modules:",len(rows))
        root_modules = [row[0]for row in rows]
        print(f"Found {len(root_modules)} root modules.")
        return root_modules

    def run_import_for_all_submodules(self, root_module: str):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON3 SCALAR SCRIPT import_all_modules.import_for_all_submodules(root_module_name VARCHAR(200000)) 
            EMITS (module_name VARCHAR(200000), exception_str VARCHAR(200000), status VARCHAR(10)) AS
            
            import sys
            import os
            import pkgutil
            import importlib
            from typing import List

            module_limit = 10000
            error_limit = 100
            excluded_modules = {
                "turtle",
                "py310compat",
                "urllib3.contrib.socks",
                "urllib3.contrib.securetransport",
                "urllib3.contrib.ntlmpool",
                "urllib3.contrib.emscripten",
                "simplejson.ordered_dict",
                "pymemcache.test",
                "sagemaker.feature_store.feature_processor",
                "sagemaker.remote_function.runtime_environment.spark_app",
                "pyftpdlib.test",
                "paramiko.win_pageant",
                "pandas.core.arrays.arrow",
                "msal_extensions.windows",
                "msal_extensions.osx",
                "msal_extensions.libsecret",
                "lxml.usedoctest",
                "lxml.html.usedoctest",
                "lxml.html.soupparser",
                "lxml.html.html5parser",
                "lxml.html.ElementSoup",
                "lxml.html.clean",
                "lxml.cssselect",
                "jsonschema.benchmarks",
                "numpy.f2py.setup",
                "numpy.distutils.msvc9compiler",
                "numpy.core.setup_common",
                "numpy.core.setup",
                "numpy.core.generate_numpy_api",
                "numpy.core.cversions",
                "numpy.core.umath_tests",
                "multiprocess.popen_spawn_win32",
                "msal.broker",
                "joblib.externals.loky.backend.popen_loky_win32",
                "ijson.backends.yajl2_cffi",
                "ijson.backends.yajl2",
                "ijson.backends.yajl",
                "docutils.parsers",
                "dateutil.tzwin",
                "dateutil.tz.win",
                "botocore.docs",
                "boto.roboto.awsqueryrequest",
                "boto.roboto.awsqueryservice",
                "boto.s3.resumable_download_handler",
                "boto.manage.test_manage",
                "boto.gs.resumable_upload_handler",
                "boto.mashups.order",
                "boto.pyami.copybot",
                "boto.requestlog",
                "boto.gs.resumable_upload_handler",
                "sagemaker.content_types",
                "pyarrow.libarrow_python_flight",
                "pyarrow.libarrow_python",
                "pyarrow.libarrow_python_parquet_encryption",
                "pyarrow.cuda",
                "numba.testing.notebook",
                "numba.np.ufunc.tbbpool",
                "numba.misc.gdb_print_extension",
                "numba.misc.dump_style",
                "martian.testing_compat3",
                "llvmlite.binding.libllvmlite",
                "docutils.writers.odf_odt.pygmentsformatter",
                "debugpy.launcher.winapi",
                "bitsets.visualize",
                "aiohttp.worker",
                "test.libregrtest.win_utils",
                "multiprocessing.popen_spawn_win32",
                "lib2to3.pgen2.conv",
                "encodings.oem",
                "encodings.mbcs",
                "distutils.msvc9compiler",
                "dbm.gnu",
                "asyncio.windows_utils",
                "asyncio.windows_events",
                "Cython.Debugger",
                "Cython.Build.Tests",
                "Cython.Build.IpythonMagic",
                "Cython.Coverage",
                "setuptools.modified",
                "tqdm.tk",
                "tqdm.rich",
                "tqdm.keras",
                "tqdm.dask",
                "tqdm.contrib.slack",
                "tqdm.contrib.discord",
                "sagemaker.serve.validations.parse_registry_accounts",
                "pyparsing.diagram",
                "numba.core.rvsdg_frontend",
                "msrest.universal_http.aiohttp",
                "msrest.pipeline.aiohttp",
                "docker.transport.npipesocket",
                "docker.transport.npipeconn"
            }
            excluded_submodules = (
                "sphinxext",
                "tests",
                "conftest",
            )
            
            def get_module_names(path):
                modules = list(module for _, module, _ in pkgutil.iter_modules(path))
                return modules

            class Importer:
                
                def __init__(
                    self, ctx, error_limit: int, module_limit: int, 
                    excluded_modules: List[str], excluded_submodules: List[str]):

                    self.ctx = ctx
                    self.error_count = 0
                    self.module_count = 0
                    self.error_limit = error_limit
                    self.module_limit = module_limit
                    self.excluded_modules = excluded_modules
                    self.excluded_submodules = excluded_submodules
                    self.modules = []

                def module_limit_reached(self) -> bool:
                    self.module_count += 1                                                                                          
                    return self.module_count > self.module_limit
                
                def error_limit_reached(self) -> bool:
                    self.error_count += 1                                                                                          
                    return self.error_count > self.error_limit

                def add_submodules(self, module, module_import):
                    if hasattr(module_import,"__path__"):
                        submodule_names = get_module_names(module_import.__path__)
                        self.modules.extend([
                            f"{module}.{submodule}"
                            for submodule in submodule_names 
                            if not submodule.startswith("_") and submodule not in self.excluded_submodules
                        ])

                def import_modules(self, root_module: str):
                    self.modules = [root_module]
                    while len(self.modules) > 0:
                        module = self.modules.pop()
                        limit_reached = self.import_module(module)
                        if limit_reached:
                            break
            
                def import_module(self, module: str) -> bool:
                    print("========================================================")
                    print("========================================================")
                    print("========================================================")
                    print("modules left: ", len(self.modules))
                    print("current module: ", module)
                    print("========================================================")
                    print("========================================================")
                    print("========================================================", flush=True)
                    if module in self.excluded_modules or module.startswith("_"):
                        self.ctx.emit(module, None, "SKIPPED")
                        return self.module_limit_reached()
                    try:
                        module_import = importlib.import_module(module)
                        self.ctx.emit(module, None, "OK")
                        self.add_submodules(module=module, module_import=module_import)
                        return self.module_limit_reached()
                    except BaseException as e:
                        import traceback
                        if hasattr(e,"msg") and e.msg == "No module named 'pytest'":
                            self.ctx.emit(module, None, "IGNORED")
                            return self.module_limit_reached()
                        else:
                            self.ctx.emit(module, traceback.format_exc(), "ERROR")
                            return self.error_limit_reached()

            def run(ctx):
                importer = Importer(
                    ctx=ctx, 
                    error_limit = error_limit,
                    module_limit = module_limit,                                                                     
                    excluded_modules = excluded_modules,
                    excluded_submodules = excluded_submodules)
                importer.import_modules(ctx.root_module_name)
            /
            '''))
        #with UdfDebugger(test_case=self):
        rows = self.query(f'''SELECT import_all_modules.import_all_submodules({root_module}) FROM dual''')
        print("Number of modules:",len(rows))
        failed_imports = [(row[0],row[1]) for row in rows if row[2] == "ERROR"]
        for i in failed_imports:
            print(i[0])
        for i in failed_imports:
            print(i[0], i[1])
        self.assertEqual(failed_imports,[])

    def test_import_all_modules(self):
        root_modules = self.get_all_root_modules()
        for root_module in root_modules:
            self.run_import_for_all_submodules(root_module)

    def tearDown(self):
        self.query("drop schema import_all_modules cascade", ignore_errors=True)


if __name__ == '__main__':
    udf.main()
