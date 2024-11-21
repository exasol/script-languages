#!/usr/bin/env python3
from typing import List, Tuple

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf.udf_debug import UdfDebugger

class ImportAllModulesTest(udf.TestCase):

    def setUp(self):
        self.query('create schema import_all_r_modules', ignore_errors=True)

    def get_all_root_modules(self) -> List[Tuple[str, str]]:
        self.query(udf.fixindent('''
            CREATE OR REPLACE r SCALAR SCRIPT import_all_r_modules.get_all_root_modules() 
            EMITS (module_name VARCHAR(200000), version VARCHAR(200)) AS
                run <- function(ctx) {
                    library(data.table)
                    file_pattern <- "cran_packages"
                    directory <- "/build_info/packages"
                    files <- list.files(path = directory, pattern = file_pattern, full.names = TRUE, recursive = TRUE)
                    for (file in files) {
                        package_list <- fread(file, sep="|", header = FALSE, col.names = c("Package", "Version"))
                        package_names <- package_list[[1]]
                        versions <- package_list[[2]]
                        ctx$emit(package_names, versions)
                    }
                }
            /
            '''))
        rows = self.query('''SELECT import_all_r_modules.get_all_root_modules() FROM dual''')
        print("Number of modules:",len(rows))
        root_modules = [(row[0], row[1]) for row in rows]
        print(f"Found {len(root_modules)} root modules.")
        return root_modules

    def create_check_installed_package_udf(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE r SCALAR SCRIPT
                import_all_r_modules.check_installed_package(package_name VARCHAR(200000), version VARCHAR(200))
            RETURNS DECIMAL(11,0) AS
            run <- function(ctx) {
             library(ctx$package_name, character.only = TRUE)
             desc <- packageDescription(ctx$package_name)
             if (ctx$version != desc$Version) {
                stop(paste("Version of  installed installed package does not match:", ctx$package_name))
                return(1)
             }
             0
            }
            /
            '''))

    def test_import_all_modules(self):
        root_modules = self.get_all_root_modules()
        assert len(root_modules) > 0
        self.create_check_installed_package_udf()
        for root_module in root_modules:
            # with UdfDebugger(test_case=self):
            rows = self.query(f'''SELECT import_all_r_modules.check_installed_package('{root_module[0]}', '{root_module[1]}') FROM dual''')

    def tearDown(self):
        self.query("drop schema import_all_r_modules cascade", ignore_errors=True)


if __name__ == '__main__':
    udf.main()
