from exaslct_src.lib.docker.docker_image_creator_base_task import DockerImageCreatorBaseTask
from exaslct_src.lib.docker.docker_image_target import DockerImageTarget
from exaslct_src.lib.docker_config import source_docker_repository_config


class DockerPullImageTask(DockerImageCreatorBaseTask):

    def run_task(self):
        image_target = DockerImageTarget(image_name=self.image_info.source_repository_name,
                                         image_tag=self.image_info.get_source_complete_tag())
        self.logger.info("Try to pull docker image %s", image_target.get_complete_name())
        if source_docker_repository_config().username is not None and \
                source_docker_repository_config().password is not None:
            auth_config = {
                "username": source_docker_repository_config().username,
                "password": source_docker_repository_config().password
            }
        else:
            auth_config = None
        self.client.images.pull(repository=image_target.image_name, tag=image_target.image_tag,
                                auth_config=auth_config)
        self.client.images.get(self.image_info.get_source_complete_name()).tag(
            repository=self.image_info.target_repository_name,
            tag=self.image_info.get_target_complete_tag()
        )
