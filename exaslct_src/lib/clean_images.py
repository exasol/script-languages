import logging
from collections import deque

import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.docker_config import docker_client_config, target_docker_repository_config
from exaslct_src.lib.flavor_task import FlavorBaseTask
from exaslct_src.lib.utils.docker_utils import find_images_by_tag


# TODO remove only images that are not represented by current flavor directories
# TODO requires that docker build only returns the image_info without actually building or pulling
class CleanExaslcFlavorImages(FlavorBaseTask):

    def register_required(self):
        flavor_name = self.get_flavor_name()

        if target_docker_repository_config().repository_name == "":
            raise Exception("docker repository name must not be an empty string")

        flavor_name_extension = ":%s" % flavor_name
        self.starts_with_pattern = target_docker_repository_config().repository_name + \
                                   flavor_name_extension
        task = self.create_child_task(CleanImagesStartingWith, starts_with_pattern=self.starts_with_pattern)
        self.register_dependency(task)

    def run_task(self):
        pass


class CleanExaslcAllImages(DependencyLoggerBaseTask):

    def register_required(self):
        self.starts_with_pattern = target_docker_repository_config().repository_name
        task = self.create_child_task(CleanImagesStartingWith, starts_with_pattern=self.starts_with_pattern)
        self.register_dependency(task)

    def run_task(self):
        pass


class CleanImagesStartingWith(DependencyLoggerBaseTask):
    starts_with_pattern = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = docker_client_config().get_client()
        self._low_level_client = docker_client_config().get_low_level_client()

    def __del__(self):
        self._client.close()

    def run_task(self):
        filter_images = self.find_images_to_clean()
        queue = deque(filter_images)
        while len(queue) != 0:
            image = queue.pop()
            image_id = image.id
            try:
                self.try_to_remove_image(image_id)
            except Exception as e:
                self.handle_errors(e, image, image_id, queue)

    def find_images_to_clean(self):
        self.logger.info("Going to remove all images starting with %s" % self.starts_with_pattern)
        filter_images = find_images_by_tag(self._client, lambda tag: tag.startswith(self.starts_with_pattern))
        for i in filter_images:
            self.logger.info("Going to remove following image: %s" % i.tags)
        return filter_images

    def try_to_remove_image(self, image_id):
        self._client.images.remove(image=image_id, force=True)
        self.logger.info("Removed image %s" % image_id)

    def handle_errors(self, e, image, image_id, queue):
        if not "No such image" in str(e) and \
                not "image is being used by running container" in str(e):
            self.add_dependent_images(image, image_id, queue)
        if "image is being used by running container" in str(e):
            self.logger.error("Unable to clean image %s, because got exception %s", image_id, e)

    def add_dependent_images(self, image, image_id, queue):
        queue.append(image)
        nb_childs = 0
        for possible_child in self._client.images.list(all=True):
            inspect = self._low_level_client.inspect_image(image=possible_child.id)
            if inspect["Parent"] == image_id:
                queue.append(possible_child)
                nb_childs = nb_childs + 1
