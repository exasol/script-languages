import logging
from typing import Dict

import docker
import luigi

from build_utils.build_config import build_config
from build_utils.docker_config import docker_config
from build_utils.build_context_hasher import BuildContextHasher

from build_utils.docker_image_builder import DockerImageBuilder
from build_utils.docker_image_target import DockerImageTarget


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
            self._build_directories_mapping, self._dockerfile)
        self._client = docker.DockerClient(base_url=self._docker_config.docker_base_url)

    def _prepare_outputs(self):
        self._image_name_target = luigi.LocalTarget(
            "%s/image_names/%s"
            % (self._build_config.ouput_directory, self._image_tag))
        if self._image_name_target.exists():
            self._image_name_target.remove()

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
        pass

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

    def _get_complete_name(self):
        complete_name = f"{self._image_name}:{self._image_tag}"
        return complete_name

    def output(self):
        return {"image_name_target": self._image_name_target}

    def run(self):
        images_names_of_dependencies = self._get_image_names_of_dependencies()
        image_hash = self._build_context_hasher.generate_image_hash(images_names_of_dependencies)
        complete_tag = self._image_tag + "_" + image_hash
        image_target = DockerImageTarget(self._image_name, complete_tag)

        if self._build_config.force_build \
                or self._build_config.force_pull:
            if image_target.exists():
                self._client.images.remove(image=image_target.get_complete_name(), force=True)
                self.logger.info("Task %s: Removed docker images %s", self.task_id, image_target.get_complete_name())
        if not image_target.exists():
            if not self._build_config.force_build \
                    and self._is_image_in_registry(image_target):
                self._pull_image(image_target)
            else:
                self._image_builder.build(image_target, images_names_of_dependencies)
        image_name_file = self.output()["image_name_target"]
        with image_name_file.open("wt") as image_name_file:
            image_name_file.write(image_target.get_complete_name())

    def _get_image_names_of_dependencies(self) -> Dict[str, str]:
        """
        Reads from input the image names produced by dependent tasks.
        :return Dictionary with dependency keys defined by requires method and images names as values
        """
        if isinstance(self.input(), Dict):
            return {key: self.get_image_name_of_depedency(value)
                    for key, value in self.input().items()
                    if isinstance(value, Dict) and "image_name_target" in value}
        else:
            return dict()

    def get_image_name_of_depedency(self, value):
        with value["image_name_target"].open("r") as file:
            return file.read()

    def _pull_image(self, image_target: DockerImageTarget):
        print("execute pull", self.task_id)
        self._client.images.pull(image_target.get_complete_name())

    def _is_image_in_registry(self, image_target: DockerImageTarget):
        try:
            registry_data = self._client.images.get_registry_data(image_target.get_complete_name())
            return True
        except docker.errors.APIError as e:
            print("Exception while checking if image exists in registry",
                  image_target.get_complete_name(), e)
            return False
