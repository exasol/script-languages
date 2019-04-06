import logging
from datetime import datetime

import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.data.dependency_collector.dependency_release_info_collector import RELEASE_INFO
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.release_type import ReleaseType
from build_utils.stoppable_task import StoppableTask


class UploadContainerTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/uploads/%s/%s/%s"
            % (self._build_config.output_directory,
               flavor.get_name_from_path(self.flavor_path),
               self.get_release_type().name,
               datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return {RELEASE_INFO: self._target}

    def requires(self):
        return self.get_export_task(self.flavor_path)

    def get_export_task(self, flavor_path):
        pass

    def get_release_type(self) -> ReleaseType:
        pass

    def run_task(self):
        pass