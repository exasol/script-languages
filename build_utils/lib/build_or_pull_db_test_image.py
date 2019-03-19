from build_utils.lib.docker.docker_pull_or_build_image_tasks import DockerPullOrBuildImageTask
from build_utils.lib.docker_config import docker_config


class BuildOrPullDBTestImage(DockerPullOrBuildImageTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()

    def get_image_name(self):
        return f"""{self._docker_config.repository_user}/{self._docker_config.repository_name}-db-test-container"""

    def get_image_tag(self):
        return "latest"

    def get_build_directories_mapping(self):
        return {"tests": "tests", "ext":"ext"}

    def get_dockerfile(self):
        return "tests/Dockerfile"