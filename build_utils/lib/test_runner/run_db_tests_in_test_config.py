import pathlib
from typing import TextIO

import luigi
from luigi import LocalTarget

from build_utils.lib.test_runner.run_db_test import RunDBTest
from build_utils.lib.test_runner.run_db_test_in_directory import RunDBTestsInDirectory


class RunDBTestsInTestConfig(luigi.Task):
    flavor_name = luigi.Parameter()
    release_type = luigi.Parameter()
    generic_language_tests = luigi.ListParameter()
    test_folders = luigi.ListParameter()
    test_files = luigi.ListParameter()
    tests_to_execute = luigi.ListParameter([])
    languages = luigi.ListParameter([None])
    environment = luigi.DictParameter({"TRAVIS": ""}, significant=False)
    language_definition = luigi.Parameter(significant=False)

    log_path = luigi.Parameter(significant=False)
    log_file_name = luigi.Parameter(significant=False)
    log_level = luigi.Parameter("critical", significant=False)
    test_environment_info_dict = luigi.DictParameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_outputs()

    def _prepare_outputs(self):
        path = pathlib.Path(self.log_path).joinpath(self.log_file_name)
        self._log_target = luigi.LocalTarget(str(path))
        # if self._log_target.exists():
        #     self._log_target.remove()

    def output(self):
        return self._log_target

    def run(self):
        with self.output().open("w") as output_file:
            yield from self.run_generic_tests(output_file)
            yield from self.run_test_folders(output_file)
            yield from self.run_test_files(output_file)

    def run_generic_tests(self, output_file):
        for language in self.generic_language_tests:
            log_path = pathlib.Path(self.log_path).joinpath("generic_tests").joinpath(language)
            test_output = yield RunDBTestsInDirectory(
                flavor_name=self.flavor_name,
                release_type=self.release_type,
                log_path=str(log_path),
                log_file_name="summary.log",
                language_definition=self.language_definition,
                language=language,
                test_environment_info_dict=self.test_environment_info_dict,
                log_level=self.log_level,
                environment=self.environment,
                tests_to_execute=self.tests_to_execute,
                directory="generic"
            )
            self.write_output("generic_tests", language, output_file, test_output)

    def run_test_folders(self, output_file: TextIO):
        for language in self.languages:
            for test_folder in self.test_folders:
                log_path = pathlib.Path(self.log_path).joinpath("test_folders")
                log_path = log_path.joinpath(test_folder)
                if language is not None:
                    log_path = log_path.joinpath(language)
                test_output = yield RunDBTestsInDirectory(
                    flavor_name=self.flavor_name,
                    release_type=self.release_type,
                    log_path=str(log_path),
                    log_file_name="summary.log",
                    language_definition=self.language_definition,
                    test_environment_info_dict=self.test_environment_info_dict,
                    log_level=self.log_level,
                    environment=self.environment,
                    tests_to_execute=self.tests_to_execute,
                    directory=test_folder
                )
                test_name = test_folder
                if language is not None:
                    test_folder += " " + language
                self.write_output("test_folder", test_name, output_file, test_output)

    def run_test_files(self, output_file: TextIO):
        for language in self.languages:
            for test_file in self.test_files:
                log_path = pathlib.Path(self.log_path).joinpath("test_files")
                log_path = log_path.joinpath(test_file)
                if language is not None:
                    log_path = log_path.joinpath(language)
                log_path.joinpath("summary.log")
                test_output = yield RunDBTest(
                    flavor_name=self.flavor_name,
                    release_type=self.release_type,
                    log_path=str(log_path),
                    language_definition=self.language_definition,
                    test_environment_info_dict=self.test_environment_info_dict,
                    log_level=self.log_level,
                    environment=self.environment,
                    tests_to_execute=self.tests_to_execute,
                    test_file=test_file,
                    language=language
                )
                test_name = test_file
                if language is not None:
                    test_file += " " + language
                self.write_output("test_file", test_name, output_file, test_output)

    def write_output(self, test_type: str, test_file: str, output_file: TextIO, test_output: LocalTarget):
        with test_output.open("r") as test_output_file:
            exit_code = test_output_file.read()
        for line in exit_code.split("\n"):
            if line != "":
                output_file.write("%s %s %s\n" % (test_type, test_file, line))
