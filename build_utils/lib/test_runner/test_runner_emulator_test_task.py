import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.data.dependency_collector.dependency_collector import DependencyCollector
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor


class TestRunnerEmulatorTestTask(MyTask):
    flavor_path = luigi.Parameter()

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
            "%s/logs/test-runner/emulator-test/%s"
            % (self._build_config.output_directory,
               flavor.get_name_from_path(self.flavor_path)))
        if self._log_target.exists():
            self._log_target.remove()

    def output(self):
        return self._log_target

    def requires(self):
        return self.get_build_run_task(self.flavor_path)

    def get_build_run_task(self, flavor_path):
        pass

    def my_run(self):
        image_info_of_dependencies = DependencyCollector().get_from_sinlge_input(self.input())

