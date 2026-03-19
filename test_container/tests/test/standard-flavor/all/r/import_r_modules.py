#!/usr/bin/env python3
from typing import List, Tuple

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf.udf_debug import UdfDebugger

class ImportAllModulesTest(udf.TestCase):

    def setUp(self):
        self.query('create schema import_all_r_modules', ignore_errors=True)

    def get_all_root_modules(self) -> List[Tuple[str, str]]:
        self.query(udf.fixindent(r'''
            CREATE OR REPLACE r SCALAR SCRIPT import_all_r_modules.get_all_root_modules() 
            EMITS (module_name VARCHAR(200000), version VARCHAR(200)) AS
            run <- function(ctx) {
              library(yaml)
            
              directory <- "/build_info/packages"
              file_pattern <- "packages\\.yml$"
              files <- list.files(
                path = directory,
                pattern = file_pattern,
                full.names = TRUE,
                recursive = FALSE
              )
            
              if (!length(files)) return(invisible(NULL))
            
              # Merge by package name; if multiple versions appear, keep the first non-empty one
              merged <- new.env(parent = emptyenv())
            
              add_pkg <- function(name, version) {
                if (is.null(name) || !nzchar(name)) return()
                if (is.null(version) || is.na(version)) version <- ""
            
                if (!exists(name, envir = merged, inherits = FALSE)) {
                  assign(name, version, envir = merged)
                } else {
                  cur <- get(name, envir = merged, inherits = FALSE)
                  if ((is.null(cur) || !nzchar(cur)) && nzchar(version)) {
                    assign(name, version, envir = merged)
                  }
                }
              }
            
              for (f in files) {
                pkg_file <- tryCatch(yaml::read_yaml(f), error = function(e) NULL)
                if (is.null(pkg_file)) next
            
                build_steps <- pkg_file$build_steps
                if (is.null(build_steps) || !length(build_steps)) next
            
                for (bs in build_steps) {
                  phases <- bs$phases
                  if (is.null(phases) || !length(phases)) next
            
                  for (ph in phases) {
                    r_section <- ph$r
                    if (is.null(r_section)) next
            
                    r_pkgs <- r_section$packages
                    if (is.null(r_pkgs) || !length(r_pkgs)) next
            
                    for (p in r_pkgs) {
                      add_pkg(p$name, p$version)
                    }
                  }
                }
              }
            
              pkg_names <- ls(envir = merged, all.names = TRUE)
              if (!length(pkg_names)) return(invisible(NULL))
            
              versions <- vapply(pkg_names, function(n) get(n, envir = merged, inherits = FALSE), "")
              ctx$emit(pkg_names, versions)
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
