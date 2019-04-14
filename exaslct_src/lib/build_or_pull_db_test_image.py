from exaslct_src.lib.docker.docker_pull_or_build_image_tasks import DockerPullOrBuildImageTask
from exaslct_src.lib.docker_config import docker_config


class BuildOrPullDBTestContainerImage(DockerPullOrBuildImageTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()

    def get_image_name(self):
        return f"""{self._docker_config.repository_name}-db-test-container"""

    def get_image_tag(self):
        return "latest"

    def get_mapping_of_build_files_and_directories(self):
        return {"requirements.txt": "tests/requirements.txt", "ext":"ext"}

    def get_dockerfile(self):
        return "tests/Dockerfile"