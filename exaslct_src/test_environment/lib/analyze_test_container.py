from typing import Set, Dict

from exaslct_src.exaslct.lib.config.docker_config import target_docker_repository_config, \
    source_docker_repository_config
from exaslct_src.exaslct.lib.tasks.push.docker_push_parameter import DockerPushParameter
from exaslct_src.test_environment.docker.images.create.docker_build_base import DockerBuildBase
from exaslct_src.test_environment.docker.images.create.docker_image_analyze_task import DockerAnalyzeImageTask
from exaslct_src.test_environment.docker.images.push.push_task_creator_for_build_tasks import \
    PushTaskCreatorFromBuildTasks


class AnalyzeTestContainer(DockerAnalyzeImageTask):

    def get_target_repository_name(self) -> str:
        return f"""{target_docker_repository_config().repository_name}"""

    def get_source_repository_name(self) -> str:
        return f"""{source_docker_repository_config().repository_name}"""

    def get_source_image_tag(self):
        if source_docker_repository_config().tag_prefix != "":
            return f"{source_docker_repository_config().tag_prefix}_db-test-container"
        else:
            return f"db-test-container"

    def get_target_image_tag(self):
        if target_docker_repository_config().tag_prefix != "":
            return f"{target_docker_repository_config().tag_prefix}_db-test-container"
        else:
            return f"db-test-container"

    def get_mapping_of_build_files_and_directories(self):
        return {"requirements.txt": "tests/requirements.txt", "ext": "ext"}

    def get_dockerfile(self):
        return "tests/Dockerfile"

    def is_rebuild_requested(self) -> bool:
        return False


class DockerTestContainerBuildBase(DockerBuildBase):

    def get_goal_class_map(self) -> Dict[str, DockerAnalyzeImageTask]:
        goal_class_map = {"test-container": AnalyzeTestContainer()}
        return goal_class_map

    def get_default_goals(self) -> Set[str]:
        goals = {"test-container"}
        return goals

    def get_goals(self):
        goals = {"test-container"}
        return goals


class DockerTestContainerBuild(DockerTestContainerBuildBase):

    def run_task(self):
        build_tasks = self.create_build_tasks(False)
        image_infos_futures = yield from self.run_dependencies(build_tasks)
        image_infos = self.get_values_from_futures(image_infos_futures)
        self.return_object(image_infos)


class DockerTestContainerPush(DockerTestContainerBuildBase, DockerPushParameter):

    def run_task(self):
        build_tasks = self.create_build_tasks(shortcut_build=not self.push_all)
        push_task_creator = PushTaskCreatorFromBuildTasks(self)
        push_tasks = push_task_creator.create_tasks_for_build_tasks(build_tasks)
        image_infos = yield from self.run_dependencies(push_tasks)
        self.return_object(image_infos)
