from typing import List, Generator, Any

import luigi

from exaslct_src.exaslct.lib.tasks.test.run_db_test import RunDBTest
from exaslct_src.exaslct.lib.tasks.test.run_db_test_result import RunDBTestDirectoryResult, RunDBTestResult
from exaslct_src.exaslct.lib.tasks.test.run_db_tests_parameter import RunDBTestParameter
from exaslct_src.test_environment.src.lib.base.flavor_task import FlavorBaseTask
from exaslct_src.test_environment.src.lib.base.json_pickle_target import JsonPickleTarget
from exaslct_src.test_environment.src.lib.data.database_credentials import DatabaseCredentialsParameter


class RunDBTestsInDirectory(FlavorBaseTask,
                            RunDBTestParameter,
                            DatabaseCredentialsParameter):
    directory = luigi.Parameter()

    def extend_output_path(self):
        return self.caller_output_path + (self.directory,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_container_info = self.test_environment_info.test_container_info
        self.tasks = self.create_test_tasks_from_directory(self.directory)

    def run_task(self):
        test_results = yield from self.run_tests()
        result = RunDBTestDirectoryResult(test_results=test_results,
                                          language=self.language,
                                          test_folder=self.directory)
        JsonPickleTarget(self.get_output_path().joinpath("test_results.json")).write(test_results, 4)
        self.return_object(result)

    def run_tests(self) -> Generator[RunDBTest, Any, List[RunDBTestResult]]:
        test_results = []
        for test_task_config in self.tasks:
            test_result_future = yield from self.run_dependencies(test_task_config)
            test_result = self.get_values_from_future(test_result_future)
            test_results.append(test_result)
        return test_results

    def create_test_tasks_from_directory(
            self, directory: str):
        test_container = self._client.containers.get(self._test_container_info.container_name)
        exit_code, ls_output = test_container.exec_run(cmd="ls /tests/test/%s/" % directory)
        test_files = ls_output.decode("utf-8").split("\n")
        tasks = [self.create_test_task(directory, test_file)
                 for test_file in test_files
                 if test_file != "" and test_file.endswith(".py")]
        return tasks

    def create_test_task(self, directory: str, test_file: str):
        task = self.create_child_task_with_common_params(
            RunDBTest,
            test_file=directory + "/" + test_file
        )
        return task