from typing import Any, Generator

from exaslct_src.lib.base.json_pickle_target import JsonPickleTarget
from exaslct_src.lib.flavor_task import FlavorBaseTask
from exaslct_src.lib.test_runner.database_credentials import DatabaseCredentialsParameter
from exaslct_src.lib.test_runner.run_db_test_in_directory import RunDBTestsInDirectory
from exaslct_src.lib.test_runner.run_db_test_result import RunDBTestFoldersResult, RunDBTestDirectoryResult
from exaslct_src.lib.test_runner.run_db_tests_parameter import RunDBGenericLanguageTestParameter, \
    ActualRunDBTestParameter


class RunDBGenericLanguageTest(FlavorBaseTask,
                               RunDBGenericLanguageTestParameter,
                               ActualRunDBTestParameter,
                               DatabaseCredentialsParameter):

    def extend_output_path(self):
        return self.caller_output_path + ("generic",)

    def run_task(self):
        results = []
        for language in self.generic_language_tests:
            test_result = yield from self.run_test(language, "generic")
            results.append(test_result)
        test_results = RunDBTestFoldersResult(test_results=results)
        JsonPickleTarget(self.get_output_path().joinpath("test_results.json")).write(test_results, 4)
        self.return_object(test_results)

    def run_test(self, language: str, test_folder: str) -> \
            Generator[RunDBTestsInDirectory, Any, RunDBTestDirectoryResult]:
        task = self.create_child_task_with_common_params(
            RunDBTestsInDirectory,
            language=language,
            directory=test_folder,
        )
        test_result_future = yield from self.run_dependencies(task)
        test_result = self.get_values_from_future(test_result_future)
        return test_result
