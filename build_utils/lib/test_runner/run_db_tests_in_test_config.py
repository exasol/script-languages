import pathlib

import luigi

from build_utils.lib.test_runner.run_db_test_in_directory import RunDBTestsInDirectory


class RunDBTestsInTestConfig(luigi.Task):
    flavor_name = luigi.Parameter()
    release_type = luigi.Parameter()
    generic_language_tests = luigi.ListParameter()
    test_folders = luigi.ListParameter()
    tests_to_execute = luigi.ListParameter([])
    environment = luigi.DictParameter({"TRAVIS": ""})
    language_definition = luigi.Parameter()

    log_path = luigi.Parameter()
    log_file_name = luigi.Parameter()
    log_level = luigi.Parameter("critical")
    test_environment_info_dict = luigi.DictParameter()

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
        with self.output().open("w") as file:
            yield from self.run_generic_tests(file)
            yield from self.run_test_folders(file)

    def run_generic_tests(self, file):
        for generic_language in self.generic_language_tests:
            test_output = yield RunDBTestsInDirectory(
                flavor_name=self.flavor_name,
                release_type=self.release_type,
                log_path=str(pathlib.Path(self.log_path).joinpath("generic").joinpath(generic_language)),
                log_file_name="summary.log",
                language_definition=self.language_definition,
                language=generic_language,
                test_environment_info_dict=self.test_environment_info_dict,
                log_level=self.log_level,
                environment=self.environment,
                tests_to_execute=self.tests_to_execute,
                directory="generic"
            )
            self.write_output_for_generic_tests(file, generic_language, test_output)

    def write_output_for_generic_tests(self, file, generic_language, test_output):
        with test_output.open("r") as test_output_file:
            exit_code = test_output_file.read()
        for line in exit_code.split("\n"):
            if line != "":
                file.write("generic %s %s\n" % (generic_language, line))

    def run_test_folders(self, file):
        for directory in self.test_folders:
            test_output = yield RunDBTestsInDirectory(
                flavor_name=self.flavor_name,
                release_type=self.release_type,
                log_path=str(pathlib.Path(self.log_path).joinpath(directory)),
                log_file_name="summary.log",
                language_definition=self.language_definition,
                test_environment_info_dict=self.test_environment_info_dict,
                log_level=self.log_level,
                environment=self.environment,
                tests_to_execute=self.tests_to_execute,
                directory=directory
            )
            self.write_output_for_test_folders(directory, file, test_output)

    def write_output_for_test_folders(self, directory, file, test_output):
        with test_output.open("r") as test_output_file:
            exit_code = test_output_file.read()
        for line in exit_code.split("\n"):
            if line != "":
                file.write("%s %s\n" % (directory, line))
