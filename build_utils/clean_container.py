import logging
import random
from collections import deque

import docker
import luigi

from build_utils.build_config import build_config
from build_utils.docker_config import docker_config
from build_utils.docker_pull_or_build_flavor_image_task import flavor


class CleanContainer(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._low_level_client = docker.APIClient(base_url=self._docker_config.base_url)
        self._flavor_config = flavor()
        self._build_config = build_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._log_target = luigi.LocalTarget(
            "%s/logs/clean-container/%s"
            % (self._build_config.ouput_directory, self.task_id))
        if self._log_target.exists():
            self._log_target.remove()

    def __del__(self):
        self._client.close()

    def output(self):
        return self._log_target

    def run(self):
        with self._log_target.open("w") as file:
            if self._docker_config.repository == "":
                raise Exception("docker repository must not be an empty string")
            images = self._client.images.list()
            filter_images = [image for image in images
                             if len(image.tags) == 1 is not None and
                             image.tags[0].startswith(self._docker_config.repository)]
            queue = deque(filter_images)
            while len(queue)!=0:
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
                    print(e)
                    if not "No such image" in str(e):
                        file.write(str(e))
                        file.write("\n")
                        queue.append(image)
                        nb_childs = 0
                        for possible_child in self._client.images.list(all=True):
                            inspect = self._low_level_client.inspect_image(image=possible_child.id)
                            if inspect["Parent"]==image_id:
                                print("append child", possible_child.id)
                                queue.append(possible_child)
                                nb_childs=nb_childs+1
                        print("nb_childs",nb_childs)

