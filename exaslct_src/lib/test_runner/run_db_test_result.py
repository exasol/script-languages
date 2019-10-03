from pathlib import Path
from typing import List


class RunDBTestResult():
    def __init__(self,
                 test_file: str,
                 language: str,
                 is_test_ok: bool,
                 test_output_file: Path):
        self.test_output_file = str(test_output_file)
        self.test_file = str(test_file)
        self.is_ok = is_test_ok
        self.language = language


class RunDBTestCollectionResult():
    def __init__(self, language: str, test_results: List[RunDBTestResult]):
        self.language = language
        self.test_results = test_results
        self.tests_are_ok = all(test_result.is_ok for test_result in test_results)


class RunDBTestDirectoryResult(RunDBTestCollectionResult):
    def __init__(self,
                 test_folder: str,
                 language: str,
                 test_results: List[RunDBTestResult]):
        super().__init__(language, test_results)
        self.test_folder = test_folder


class RunDBTestFilesResult():
    def __init__(self,
                 test_results: List[RunDBTestCollectionResult]):
        self.test_results = test_results
        self.tests_are_ok = all(test_result.tests_are_ok for test_result in test_results)


class RunDBTestFoldersResult():
    def __init__(self,
                 test_results: List[RunDBTestDirectoryResult]):
        self.test_results = test_results
        self.tests_are_ok = all(test_result.tests_are_ok for test_result in test_results)


class RunDBTestsInTestConfigResult():
    def __init__(self,
                 flavor_path: str,
                 release_goal: str,
                 generic_language_tests_output: RunDBTestFoldersResult,
                 test_folders_output: RunDBTestFoldersResult,
                 test_files_output: RunDBTestFilesResult):
        self.release_goal = release_goal
        self.flavor_path = str(flavor_path)
        self.test_files_output = test_files_output
        self.test_folders_output = test_folders_output
        self.generic_language_tests_output = generic_language_tests_output
        self.tests_are_ok = \
            generic_language_tests_output.tests_are_ok and \
            test_folders_output.tests_are_ok and \
            test_files_output.tests_are_ok
