import pathlib

import luigi
from docker.models.containers import Container

from exaslct_src.lib.data.environment_info import EnvironmentInfo
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.test_runner.run_db_test import RunDBTest
from exaslct_src.lib.stoppable_task import StoppableTask


class RunDBTestsInDirectory(StoppableTask):
    directory = luigi.Parameter()
    flavor_name = luigi.Parameter()
    release_type = luigi.Parameter()
    language = luigi.OptionalParameter(None)
    test_restrictions = luigi.ListParameter([])
    test_environment_vars = luigi.DictParameter({"TRAVIS": ""}, significant=False)
    language_definition = luigi.Parameter(significant=False)

    log_path = luigi.Parameter(significant=False)
    log_level = luigi.Parameter("critical", significant=False)
    test_environment_info_dict = luigi.DictParameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        test_evironment_info = EnvironmentInfo.from_dict(self.test_environment_info_dict)
        self._test_container_info = test_evironment_info.test_container_info
        self._client = docker_client_config().get_client()
        self._prepare_outputs()
        self.test_container = self._client.containers.get(self._test_container_info.container_name)
        self.tasks = self.generate_test_task_configs_from_directory(self.test_container, self.directory)

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        path = pathlib.Path(self.log_path).joinpath("summary.log")
        self._summary_target = luigi.LocalTarget(str(path))
        # if self._log_target.exists():
        #     self._log_target.remove()

    def output(self):
        return self._summary_target

    def run_task(self):
        test_outputs = yield from self.run_tests()
        self.collect_test_outputs(test_outputs)

    def collect_test_outputs(self, test_outputs):
        with self.output().open("w") as file:
            for test_file, test_output in test_outputs:
                with test_output.open("r") as test_output_file:
                    status = test_output_file.read()
                file.write("%s %s\n" % (test_file, status))

    def run_tests(self):
        test_outputs = []
        for test_file, test_task_config in self.tasks:
            if self.language is not None:
                test_task_config["language"] = self.language
            test_output = yield RunDBTest(**test_task_config)
            test_outputs.append((test_file, test_output))
        return test_outputs

    def generate_test_task_configs_from_directory(
            self, test_container: Container, directory: str):
        exit_code, ls_output = test_container.exec_run(cmd="ls /tests/test/%s/" % directory)
        test_files = ls_output.decode("utf-8").split("\n")
        result = [(test_file, self.create_config_for_test(directory, test_file))
                  for test_file in test_files
                  if test_file != ""]
        return result

    def create_config_for_test(self, directory, test_file):
        config = dict(flavor_name=self.flavor_name,
                      release_type=self.release_type,
                      test_environment_info_dict=self.test_environment_info_dict,
                      language_definition=self.language_definition,
                      log_level=self.log_level,
                      test_environment_vars=self.test_environment_vars,
                      log_path=self.log_path,
                      test_restrictions=self.test_restrictions,
                      test_file=directory + "/" + test_file)
        return config
