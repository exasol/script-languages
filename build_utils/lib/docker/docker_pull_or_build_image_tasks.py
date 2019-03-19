import logging
from typing import Dict, Any

import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.utils.build_context_hasher import BuildContextHasher
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector, IMAGE_INFO
from build_utils.lib.docker_config import docker_config
from build_utils.lib.docker.docker_image_builder import DockerImageBuilder
from build_utils.lib.docker.docker_image_target import DockerImageTarget
from build_utils.lib.data.image_info import ImageInfo


class DockerPullOrBuildImageTask(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._image_name = self.get_image_name()
        self._image_tag = self.get_image_tag()
        self._build_directories_mapping = self.get_build_directories_mapping()
        self._dockerfile = self.get_dockerfile()
        self._prepare_outputs()
        self._build_context_hasher = BuildContextHasher(self._build_directories_mapping, self._dockerfile)
        self._image_builder = DockerImageBuilder(
            self.task_id, self._build_config, self._docker_config,
            self._build_directories_mapping, self._dockerfile,
            self.get_additional_docker_build_options())
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)

    def _prepare_outputs(self):
        self._image_info_target = luigi.LocalTarget(
            "%s/image_info/%s/%s"
            % (self._build_config.ouput_directory,
               self._image_name, self._image_tag))
        if self._image_info_target.exists():
            self._image_info_target.remove()

    def __del__(self):
        self._client.close()

    def get_image_name(self) -> str:
        """
        Called by the constructor to get the image name. Sub classes need to implement this method.
        :return: image name
        """
        pass

    def get_image_tag(self) -> str:
        """
        Called by the constructor to get the image tag. Sub classes need to implement this method.
        :return: image tag
        """
        return "latest"

    def get_build_directories_mapping(self) -> Dict[str, str]:
        """
        Called by the constructor to get the build directories mapping.
        The keys are the relative paths to the destination in build context and
        the values are the paths to the source directories or files.
        Sub classes need to implement this method.
        :return: dictionaries with destination path as keys and source paths in values
        """
        pass

    def get_dockerfile(self) -> str:
        """
        Called by the constructor to get the path to the dockerfile.
        Sub classes need to implement this method.
        :return: path to the dockerfile
        """
        pass

    def get_additional_docker_build_options(self) -> Dict[str, Any]:
        return {}

    def output(self):
        return {IMAGE_INFO: self._image_info_target}

    def run(self):
        image_info_of_dependencies = DependencyImageInfoCollector().get_from_dict_of_inputs(self.input())
        image_hash = self._build_context_hasher.generate_image_hash(image_info_of_dependencies)
        complete_tag = self._image_tag + "_" + image_hash
        image_target = DockerImageTarget(self._image_name, complete_tag)
        image_info = ImageInfo(
            complete_name=image_target.get_complete_name(),
            name=self._image_name, tag=self._image_tag, hash=image_hash,
            is_new=None, depends_on_images=list(image_info_of_dependencies.values()))
        is_new = self.pull_or_build(image_info_of_dependencies, image_target, image_info)
        image_info.is_new = is_new
        self.write_image_info_to_output(image_info)

    def pull_or_build(self,
                      image_info_of_dependencies: Dict[str, ImageInfo],
                      image_target: DockerImageTarget,
                      image_info: ImageInfo):
        is_new = False
        self.remove_image_if_required(image_target)
        if not image_target.exists():
            if not self._build_config.force_build and self._is_image_in_registry(image_target):
                self._pull_image(image_target)
                is_new = True
            else:
                self._image_builder.build(image_info, image_info_of_dependencies)
                is_new = True
        return is_new

    def remove_image_if_required(self, image_target):
        if self._build_config.force_build \
                or self._build_config.force_pull:
            if image_target.exists():
                self._client.images.remove(image=image_target.get_complete_name(), force=True)
                self.logger.info("Task %s: Removed docker images %s",
                                 self.task_id, image_target.get_complete_name())

    def write_image_info_to_output(self, image_info: ImageInfo):
        with  self.output()[IMAGE_INFO].open("wt") as file:
            file.write(image_info.to_json())

    def _pull_image(self, image_target: DockerImageTarget):
        self.logger.info("Task %s: Pull docker image %s", self.task_id, image_target.get_complete_name())
        self._client.images.pull(repository=image_target.image_name, tag=image_target.image_tag)

    def _is_image_in_registry(self, image_target: DockerImageTarget):
        try:
            registry_data = self._client.images.get_registry_data(image_target.get_complete_name())
            return True
        except docker.errors.APIError as e:
            self.logger.error("Task %s: Image %s not in registry, got exception %s", self.task_id,
                              image_target.get_complete_name(), e)
            return False
