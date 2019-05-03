import datetime
import logging
from collections import deque

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.flavor import flavor
from exaslct_src.lib.utils.docker_utils import find_images_by_tag
from exaslct_src.lib.stoppable_task import StoppableWrapperTask, StoppableTask


# TODO remove only images that are not represented by current flavor directories
# TODO requires that docker build only returns the image_info without actually building or pulling
class CleanExaslcImages(StoppableWrapperTask):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.OptionalParameter(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.flavor_path is None:
            self.flavor_name = None
        else:
            self.flavor_name = flavor.get_name_from_path(self.flavor_path)

        if docker_config().repository_name == "":
            raise Exception("docker repository name must not be an empty string")

        if self.flavor_name is not None:
            flavor_name_extension = ":%s" % self.flavor_name
        else:
            flavor_name_extension = ""
        self.starts_with_pattern = docker_config().repository_name + \
                                   flavor_name_extension

    def requires_tasks(self):
        return CleanImagesStartingWith(self.starts_with_pattern)


class CleanImagesStartingWith(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    starts_with_pattern = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = docker_config().get_client()
        self._low_level_client = docker_config().get_low_level_client()

        self._prepare_outputs()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/clean/images/%s_%s"
            % (build_config().output_directory,
               datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
               self.task_id))
        if self._log_target.exists():
            self._log_target.remove()

    def __del__(self):
        self._client.close()

    def output(self):
        return self._log_target

    def run_task(self):
        with self._log_target.open("w") as file:
            filter_images = self.find_images_to_clean()
            queue = deque(filter_images)
            while len(queue) != 0:
                image = queue.pop()
                image_id = image.id
                try:
                    self.try_to_remove_image(file, image_id)
                except Exception as e:
                    self.handle_errors(e, file, image, image_id, queue)

    def find_images_to_clean(self):
        self.logger.info("Going to remove all images starting with %s" % self.starts_with_pattern)
        filter_images = find_images_by_tag(self._client, lambda tag: tag.startswith(self.starts_with_pattern))
        for i in filter_images:
            self.logger.info("Going to remove following image: %s" % i.tags)
        return filter_images

    def try_to_remove_image(self, file, image_id):
        file.write("Try to remove image %s" % image_id)
        file.write("\n")
        self._client.images.remove(image=image_id, force=True)
        file.write("Removed image %s" % image_id)
        file.write("\n")
        self.logger.info("Removed image %s" % image_id)

    def handle_errors(self, e, file, image, image_id, queue):
        if not "No such image" in str(e) and \
                not "image is being used by running container" in str(e):
            self.add_dependent_images(e, file, image, image_id, queue)
        if "image is being used by running container" in str(e):
            self.logger.error("Unable to clean image %s, because got exception %s", image_id, e)

    def add_dependent_images(self, e, file, image, image_id, queue):
        file.write(str(e))
        file.write("\n")
        queue.append(image)
        nb_childs = 0
        for possible_child in self._client.images.list(all=True):
            inspect = self._low_level_client.inspect_image(image=possible_child.id)
            if inspect["Parent"] == image_id:
                queue.append(possible_child)
                nb_childs = nb_childs + 1
