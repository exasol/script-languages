from exaslct_src.lib.docker.docker_image_builder import DockerImageBuilder
from exaslct_src.lib.docker.docker_image_creator_base_task import DockerImageCreatorBaseTask


class DockerBuildImageTask(DockerImageCreatorBaseTask):

    def run_task(self):
        image_builder = DockerImageBuilder(self.task_id)
        image_builder.build(self.image_info)