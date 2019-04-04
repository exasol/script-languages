import pathlib

import docker
import luigi
from docker.models.containers import Container

from build_utils.lib.data.environment_info import EnvironmentInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.test_runner.run_db_test import RunDBTest


class RunDBTestsInDirectory(luigi.Task):
    directory = luigi.Parameter()
    flavor_name = luigi.Parameter()
    release_type = luigi.Parameter()
    language = luigi.OptionalParameter(None)
    tests_to_execute = luigi.ListParameter([])
    environment = luigi.DictParameter({"TRAVIS": ""},significant=False)
    language_definition = luigi.Parameter(significant=False)

    log_path = luigi.Parameter(significant=False)
    log_file_name = luigi.Parameter(significant=False)
    log_level = luigi.Parameter("critical",significant=False)
    test_environment_info_dict = luigi.DictParameter(significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        test_evironment_info = EnvironmentInfo.from_dict(self.test_environment_info_dict)
        self._test_container_info = test_evironment_info.test_container_info
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        path = pathlib.Path(self.log_path).joinpath(self.log_file_name)
        self._log_target = luigi.LocalTarget(str(path))
        # if self._log_target.exists():
        #     self._log_target.remove()

    def output(self):
        return self._log_target

    def run(self):
        test_container = self._client.containers.get(self._test_container_info.container_name)
        with self.output().open("w") as file:
            for test_file, test_task_config in \
                    self.generate_test_task_configs_from_directory(
                        test_container, self.directory):
                if self.language is not None:
                    test_task_config["language"] = self.language
                test_output = yield RunDBTest(**test_task_config)
                with test_output.open("r") as test_output_file:
                    exit_code = test_output_file.read()
                file.write("%s %s\n" % (test_file, exit_code))

    def generate_test_task_configs_from_directory(
            self, test_container: Container, directory: str):
        exit_code, ls_output = test_container.exec_run(cmd="ls /tests/test/%s/" % directory)
        test_files = ls_output.decode("utf-8").split("\n")
        for test_file in test_files:
            if test_file != "":
                config = dict(flavor_name=self.flavor_name,
                              release_type=self.release_type,
                              test_environment_info_dict=self.test_environment_info_dict,
                              language_definition=self.language_definition,
                              log_level=self.log_level,
                              environment=self.environment,
                              log_path=self.log_path,
                              tests_to_execute=self.tests_to_execute,
                              test_file=directory + "/" + test_file)
                yield test_file, config
