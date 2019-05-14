import logging
from pathlib import Path
from typing import Dict, Any

import docker
import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector, \
    IMAGE_INFO
from exaslct_src.lib.data.image_info import ImageInfo, ImageState, ImageDescription
from exaslct_src.lib.docker.docker_image_target import DockerImageTarget
from exaslct_src.lib.docker.docker_registry_image_checker import DockerRegistryImageChecker
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.utils.build_context_hasher import BuildContextHasher
from exaslct_src.lib.stoppable_task import StoppableTask


class DockerAnalyzeImageTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self._image_name = self.get_image_name()
        self._image_tag = self.get_image_tag()
        self.image_description = ImageDescription(
            dockerfile=self.get_dockerfile(),
            mapping_of_build_files_and_directories=self.get_mapping_of_build_files_and_directories(),
            image_changing_build_arguments=self.get_image_changing_build_arguments(),
            transparent_build_arguments=self.get_transparent_build_arguments()
        )
        self._dockerfile = self.get_dockerfile()
        self._prepare_outputs()
        self._build_context_hasher = \
            BuildContextHasher(self.__repr__(),
                               self.image_description)
        self._client = docker_config().get_client()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._image_info_target = luigi.LocalTarget(
            "%s/info/image/analyze/%s/%s"
            % (build_config().output_directory,
               self._image_name, self._image_tag))
        if self._image_info_target.exists():
            self._image_info_target.remove()

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

    def get_mapping_of_build_files_and_directories(self) -> Dict[str, str]:
        """
        Called by the constructor to get the build files and directories mapping.
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

    def get_image_changing_build_arguments(self) -> Dict[str, Any]:
        """
        Called by the constructor to get the path image changing docker build arguments.
        Different values for these arguments might change the image, such that they
        will be part of the image hash and can cause a build.
        A common use case is define a mirror from which packages should be installed.
        Sub classes need to implement this method.
        :return: Dictionary of build arguments, where the keys are the argument name
        """
        return dict()

    def get_transparent_build_arguments(self) -> Dict[str, Any]:
        """
        Called by the constructor to get the path transparent docker build arguments.
        Transparent arguments do not change the contain of the images.
        They are not part of the image hash. A common use case is define a mirror
        from which packages should be installed.
        Sub classes need to implement this method.
        :return: Dictionary of build arguments, where the keys are the argument name
        """
        return dict()

    def is_rebuild_requested(self) -> bool:
        pass

    def output(self):
        return {IMAGE_INFO:self._image_info_target}

    def run_task(self):
        image_info_of_dependencies = DependencyImageInfoCollector().get_from_dict_of_inputs(self.input())
        image_hash = self._build_context_hasher.generate_image_hash(image_info_of_dependencies)
        complete_tag = self._image_tag + "_" + image_hash
        image_target = DockerImageTarget(self._image_name, complete_tag)
        image_state = self.get_image_state(image_target, image_info_of_dependencies)
        image_info = ImageInfo(
            complete_name=image_target.get_complete_name(),
            complete_tag=complete_tag,
            name=self._image_name, tag=self._image_tag, hash=image_hash,
            depends_on_images=image_info_of_dependencies,
            image_state=image_state,
            image_description=self.image_description
        )
        with self._image_info_target.open("w") as f:
            f.write(image_info.to_json())

    def get_image_state(self, image_target: DockerImageTarget,
                        image_info_of_dependencies: Dict[str, ImageInfo]) -> ImageState:

        if self.is_rebuild_necessary(image_info_of_dependencies):
            return ImageState.NEEDS_TO_BE_BUILD
        else:
            if image_target.exists() \
                    and not build_config().force_pull \
                    and not build_config().force_load:
                self.logger.info(
                    f"Task {self.__repr__()}: Checking if image {image_target.get_complete_name()} "
                    f"is locally available, result {image_target.exists()}")
                return ImageState.LOCALLY_AVAILABLE
            elif self.can_image_be_loaded(image_target):
                return ImageState.CAN_BE_LOADED
            elif self.is_image_in_registry(image_target):
                return ImageState.REMOTE_AVAILABLE
            else:
                return ImageState.NEEDS_TO_BE_BUILD

    def needs_any_dependency_to_be_build(
            self, image_info_of_dependencies: Dict[str, ImageInfo]) -> bool:
        return any(image_info.image_state == ImageState.NEEDS_TO_BE_BUILD
                   for image_info in image_info_of_dependencies.values())

    def is_rebuild_necessary(self, image_info_of_dependencies: Dict[str, ImageInfo]):
        return self.needs_any_dependency_to_be_build(image_info_of_dependencies) or \
               self.is_rebuild_requested()

    def can_image_be_loaded(self, image_target: DockerImageTarget):
        if build_config().cache_directory is not None:
            image_path = self.get_path_to_cached_image(image_target)
            self.logger.info(f"Task {self.__repr__()}: Checking if image archive {image_path} "
                             f"is available in cache directory, "
                             f"result {image_path.exists()}")
            return image_path.exists()
        else:
            return False

    def get_path_to_cached_image(self, image_target):
        image_path = \
            Path(build_config().cache_directory) \
                .joinpath(Path(image_target.get_complete_name()+".tar"))
        return image_path

    def is_image_in_registry(self, image_target: DockerImageTarget):
        try:
            self.logger.info("Task %s: Try to find image %s in registry", self.__repr__(),
                             image_target.get_complete_name())
            exists = DockerRegistryImageChecker().check(image_target.get_complete_name())
            if exists:
                self.logger.info("Task %s: Found image %s in registry", self.__repr__(),
                                 image_target.get_complete_name())
            return exists
        except docker.errors.DockerException as e:
            self.logger.warning("Task %s: Image %s not in registry, got exception %s", self.__repr__(),
                                image_target.get_complete_name(), e)
            return False
