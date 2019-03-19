import luigi

from build_utils.lib.data.dependency_collector.dependency_container_info_collector import CONTAINER_INFO


class SpawnTestContainer(luigi.Task):
    database_info = luigi.DictParameter()
    db_test_image_info = luigi.DictParameter()
    container_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._test_container_info_target = luigi.LocalTarget(
            "%s/test-runner/db-test/test-container/%s"
            % (self._build_config.ouput_directory,
               self.container_name))
        if self._test_container_info_target.exists():
            self._database_info_target.remove()

    def output(self):
        return {CONTAINER_INFO: self._test_container_info_target}

    def run(self):
        pass
