import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.docker_config import docker_client_config, target_docker_repository_config
from exaslct_src.lib.utils.docker_utils import find_images_by_tag


class CleanImageTask(DependencyLoggerBaseTask):
    image_id = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = docker_client_config().get_client()
        self._low_level_client = docker_client_config().get_low_level_client()

    def __del__(self):
        self._client.close()
        self._low_level_client.close()

    def run_task(self):
        self.logger.info("Try to remove dependent images of %s" % self.image_id)
        yield from self.run_dependencies(self.get_clean_image_tasks_for_dependent_images())
        for i in range(3):
            try:
                self.logger.info("Try to remove image %s" % self.image_id)
                self._client.images.remove(image=self.image_id, force=True)
                self.logger.info("Removed image %s" % self.image_id)
                break
            except Exception as e:
                self.logger.info("Could not removed image %s got exception %s" % (self.image_id, e))

    def get_clean_image_tasks_for_dependent_images(self):
        image_ids = [str(possible_child).replace("sha256:", "") for possible_child
                     in self._low_level_client.images(all=True, quiet=True)
                     if self.is_child_image(possible_child)]
        return [self.create_child_task(CleanImageTask, image_id=image_id)
                for image_id in image_ids]

    def is_child_image(self, possible_child):
        try:
            inspect = self._low_level_client.inspect_image(image=str(possible_child).replace("sha256:", ""))
            return str(inspect["Parent"]).replace("sha256:", "") == self.image_id
        except:
            return False


class CleanImagesStartingWith(DependencyLoggerBaseTask):
    starts_with_pattern = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        self._client = docker_client_config().get_client()
        super().__init__(*args, **kwargs)

    def __del__(self):
        self._client.close()

    def register_required(self):
        image_ids = [str(image.id).replace("sha256:", "")
                     for image in self.find_images_to_clean()]
        self.register_dependencies([self.create_child_task(CleanImageTask, image_id=image_id)
                                  for image_id in image_ids])

    def find_images_to_clean(self):
        self.logger.info("Going to remove all images starting with %s" % self.starts_with_pattern)
        filter_images = find_images_by_tag(self._client, lambda tag: tag.startswith(self.starts_with_pattern))
        for i in filter_images:
            self.logger.info("Going to remove following image: %s" % i.tags)
        return filter_images

    def run_task(self):
        pass


# TODO remove only images that are not represented by current flavor directories
# TODO requires that docker build only returns the image_info without actually building or pulling
class CleanExaslcFlavorImages(DependencyLoggerBaseTask):

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
