from typing import Any, Generator

import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.base.json_pickle_target import JsonPickleTarget
from exaslct_src.lib.flavor_task import FlavorBaseTask
from exaslct_src.lib.test_runner.database_credentials import DatabaseCredentialsParameter
from exaslct_src.lib.test_runner.run_db_generic_language_tests import RunDBGenericLanguageTest
from exaslct_src.lib.test_runner.run_db_test_files import RunDBTestFiles
from exaslct_src.lib.test_runner.run_db_test_folder import RunDBTestFolder
from exaslct_src.lib.test_runner.run_db_test_result import RunDBTestsInTestConfigResult, RunDBTestCollectionResult, \
    RunDBTestFoldersResult, RunDBTestFilesResult
from exaslct_src.lib.test_runner.run_db_tests_parameter import RunDBTestsInTestConfigParameter, \
    ActualRunDBTestParameter


# TODO fetch database logs after test execution
class RunDBTestsInTestConfig(FlavorBaseTask,
                             RunDBTestsInTestConfigParameter,
                             ActualRunDBTestParameter,
                             DatabaseCredentialsParameter):

    def run_task(self):
        test_folders_output = yield from self.run_test_folder()
        test_files_output = yield from self.run_test_files()
        generic_language_test_output = yield from self.run_generic_language_test()
        result = RunDBTestsInTestConfigResult(flavor_path=self.flavor_path,
                                              release_goal=self.release_goal,
                                              generic_language_tests_output=generic_language_test_output,
                                              test_folders_output=test_folders_output,
                                              test_files_output=test_files_output)
        JsonPickleTarget(self.get_output_path().joinpath("test_results.json")).write(result,4)
        self.return_object(result)

    def run_generic_language_test(self) -> \
            Generator[RunDBGenericLanguageTest, Any, RunDBTestFoldersResult]:
        generic_language_test_task = self.create_child_task_with_common_params(RunDBGenericLanguageTest)
        generic_language_test_output_future = yield from self.run_dependencies(generic_language_test_task)
        return self.get_values_from_future(generic_language_test_output_future)

    def run_test_files(self) -> \
            Generator[RunDBGenericLanguageTest, Any, RunDBTestFilesResult]:
        test_files_task = self.create_child_task_with_common_params(RunDBTestFiles)
        test_files_output_future = yield from self.run_dependencies(test_files_task)
        return self.get_values_from_future(test_files_output_future)

    def run_test_folder(self) -> \
            Generator[RunDBGenericLanguageTest, Any, RunDBTestFoldersResult]:
        test_folder_task = self.create_child_task_with_common_params(RunDBTestFolder)
        test_folder_output_future = yield from self.run_dependencies(test_folder_task)
        return self.get_values_from_future(test_folder_output_future)
