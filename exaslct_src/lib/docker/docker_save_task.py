import logging
import os
import pathlib
import re
from typing import Generator

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from exaslct_src.lib.data.image_info import ImageInfo, ImageState
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.log_config import log_config
from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.lib.stoppable_task import StoppableTask

DOCKER_HUB_REGISTRY_URL_REGEX = r"^.*docker.io/"

# TODO align and extract save_path of DockerSaveImageTask and load_path of DockerLoadImageTask
class DockerSaveImageBaseTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    image_name = luigi.Parameter()
    force_save = luigi.BoolParameter(False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)
    save_path = luigi.Parameter(visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = docker_config().get_client()
        self._log_config = log_config()
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._save_info_target = luigi.LocalTarget(
            "%s/info/save/%s"
            % (build_config().output_directory, self.task_id))
        if self._save_info_target.exists():
            self._save_info_target.remove()

    def output(self):
        return self._save_info_target

    def requires_tasks(self):
        return self.get_docker_image_task()

    def get_docker_image_task(self):
        pass

    def run_task(self):
        image_info = DependencyImageInfoCollector().get_from_sinlge_input(self.input())
        tag_for_save = self.get_tag_for_save(image_info)
        save_file_path = pathlib.Path("%s/%s.tar" % (self.save_path, image_info.complete_name))
        was_build = image_info.image_state == ImageState.WAS_BUILD.name
        if was_build or self.force_save or not save_file_path.exists():
            self.save_image(image_info, tag_for_save, save_file_path)
        self.write_output(image_info)

    def get_tag_for_save(self, image_info):
        tag_for_save = re.sub(DOCKER_HUB_REGISTRY_URL_REGEX,"", image_info.complete_name)
        return tag_for_save

    def save_image(self, image_info: ImageInfo, tag_for_save: str, save_file_path: pathlib.Path):
        self.remove_save_file_if_necassary(save_file_path)
        image = self._client.images.get(image_info.complete_name)
        generator = image.save(named=tag_for_save)
        self.write_image_to_file(save_file_path, image_info, generator)

    def remove_save_file_if_necassary(self, save_file_path: pathlib.Path):
        save_file_path.parent.mkdir(exist_ok=True, parents=True)
        if save_file_path.exists():
            os.remove(save_file_path)

    def write_image_to_file(self,
                            save_file_path: pathlib.Path,
                            image_info: ImageInfo,
                            output_generator: Generator):
        self.logger.info(f"Task {self.__repr__()}: Saving image {image_info.complete_name} to file {save_file_path}")
        with save_file_path.open("wb") as file:
            still_running_logger = StillRunningLogger(
                self.logger, self.__repr__(), "save image %s" % image_info.complete_name)
            for chunk in output_generator:
                still_running_logger.log()
                file.write(chunk)

    def write_output(self, image_info):
        with self._save_info_target.open("w") as file:
            file.write(image_info.complete_name)
