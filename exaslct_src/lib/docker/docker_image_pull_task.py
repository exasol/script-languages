from exaslct_src.lib.docker.docker_image_creator_base_task import DockerImageCreatorBaseTask
from exaslct_src.lib.docker.docker_image_target import DockerImageTarget
from exaslct_src.lib.docker_config import docker_config


class DockerPullImageTask(DockerImageCreatorBaseTask):

    def run_task(self):
        image_target = DockerImageTarget(image_name=self.image_info.name,
                                         image_tag=self.image_info.complete_tag)
        self.logger.info("Task %s: Try to pull docker image %s", self.__repr__(), image_target.get_complete_name())
        if docker_config().username is not None and \
                docker_config().password is not None:
            auth_config = {
                "username": docker_config().username,
                "password": docker_config().password
            }
        else:
            auth_config = None
        self.client.images.pull(repository=image_target.image_name, tag=image_target.image_tag,
                                auth_config=auth_config)