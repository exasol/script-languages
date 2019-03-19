import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.data.dependency_collector.dependency_database_info_collector import DependencyDatabaseInfoCollector
from build_utils.lib.data.dependency_collector.dependency_release_info_collector import DependencyReleaseInfoCollector
from build_utils.lib.data.dependency_collector.dependency_container_info_collector import DependencyContainerInfoCollector
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.lib.test_runner.spawn_test_environment import SpawnTestEnvironment
from build_utils.lib.test_runner.upload_release_container import UploadReleaseContainer


class TestRunnerDBTestTask(luigi.Task):
    flavor_path = luigi.Parameter()
    tests_to_execute = luigi.ListParameter([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._build_config = build_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/test-runner/db-test/%s/%s/"
            % (self._build_config.ouput_directory,
               flavor.get_name_from_path(self.flavor_path),
               self.get_release_type()))
        if self._log_target.exists():
            self._log_target.remove()

    def output(self):
        return self._log_target

    def requires(self):
        test_environment_name = f"""{flavor.get_name_from_path(self.flavor_path)}_{self.get_release_type()}"""
        return {
            "release": self.get_release_task(self.flavor_path),
            "test_environment": SpawnTestEnvironment(environment_name=test_environment_name),
        }

    def get_release_task(self, flavor_path):
        pass

    def get_release_type(self):
        pass

    def run(self):
        release_info_of_dependencies = \
            DependencyReleaseInfoCollector().get_from_dict_of_inputs(self.input())
        release_info = release_info_of_dependencies["release"]
        database_info_of_dependencies = \
            DependencyDatabaseInfoCollector().get_from_dict_of_inputs(self.input())
        database_info = database_info_of_dependencies["test_environment"]
        container_info_of_dependencies = \
            DependencyContainerInfoCollector().get_from_dict_of_inputs(self.input())
        container_info = container_info_of_dependencies["test_environment"]
        test_config = self.read_test_config()
        yield UploadReleaseContainer(database_info=database_info.to_json())

    def read_test_config(self):
        pass
