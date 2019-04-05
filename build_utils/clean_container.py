import datetime
import logging
from collections import deque

import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.stoppable_task import StoppableTask


class CleanImages(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.OptionalParameter(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._low_level_client = docker.APIClient(base_url=self._docker_config.base_url)
        if self.flavor_path is None:
            self.flavor_name = None
        else:
            self.flavor_name = flavor.get_name_from_path(self.flavor_path)
        self._build_config = build_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/clean-container/%s_%s"
            % (self._build_config.output_directory,
               datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
               self.task_id))
        if self._log_target.exists():
            self._log_target.remove()

    def __del__(self):
        self._client.close()

    def output(self):
        return self._log_target

    def my_run(self):
        with self._log_target.open("w") as file:
            if self._docker_config.repository_name == "":
                raise Exception("docker repository name must not be an empty string")
            images = self._client.images.list()
            if self.flavor_name is not None:
                flavor_name_extension = ":%s" % self.flavor_name
            else:
                flavor_name_extension = ""
            starts_with_pattern = self._docker_config.repository_name + \
                                  flavor_name_extension
            self.logger.info("Going to remove all images starting with %s"%starts_with_pattern)
            filter_images = [image for image in images
                             if len(image.tags) >= 1 is not None and
                             any([tag.startswith(starts_with_pattern) for tag in image.tags])]
            self.logger.info("Going to remove following images %s" % filter_images)
            queue = deque(filter_images)
            while len(queue) != 0:
                image = queue.pop()
                image_id = image.id
                try:
                    file.write("Try to remove image %s" % image_id)
                    file.write("\n")
                    self._client.images.remove(image=image_id, force=True)
                    file.write("Removed image %s" % image_id)
                    file.write("\n")
                    self.logger.info("Removed image %s" % image_id)
                except Exception as e:
                    if not "No such image" in str(e):
                        file.write(str(e))
                        file.write("\n")
                        queue.append(image)
                        nb_childs = 0
                        for possible_child in self._client.images.list(all=True):
                            inspect = self._low_level_client.inspect_image(image=possible_child.id)
                            if inspect["Parent"] == image_id:
                                queue.append(possible_child)
                                nb_childs = nb_childs + 1
