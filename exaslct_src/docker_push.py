import importlib

from exaslct_src.docker_build import *
from exaslct_src.lib.build_or_pull_db_test_image import BuildOrPullDBTestContainerImage
from exaslct_src.lib.docker.docker_push_task import DockerPushImageTask

# We need to create the ImageBuildTask for ImagePushTask dynamically,
# because we want to push as soon as possible after an image was build and
# don't want to wait for the push finishing before starting the build of depended images,
# but we also need to create a ImagePushTask for each ImageBuildTask of a goal

class DockerPush_BuildTaskFromClassName(DockerPushImageTask):

    docker_image_task_class = luigi.Parameter()
    docker_image_task_module = luigi.Parameter()
    docker_image_task_params = luigi.DictParameter(visibility=luigi.parameter.ParameterVisibility.PRIVATE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self):
        module = importlib.import_module(self.docker_image_task_module)
        class_ = getattr(module, self.docker_image_task_class)
        instance = class_(**self.docker_image_task_params)
        return instance


class DockerPush(FlavorWrapperTask):
    force_push = luigi.BoolParameter(False)
    goals = luigi.ListParameter([])

    def requires(self):
        build_wrapper_task = DockerBuild(flavor_paths=self.actual_flavor_paths, goals=self.goals)
        build_tasks_per_flavor = build_wrapper_task.requires()
        pull_tasks = [self.generate_tasks_for_flavor(build_tasks)
                      for build_tasks in build_tasks_per_flavor]
        pull_tasks.append(self.create_push_task(BuildOrPullDBTestContainerImage()))
        return pull_tasks

    def generate_tasks_for_flavor(self, build_tasks: Set[DockerPullOrBuildImageTask]):
        return [self.create_push_task(build_task)
                for build_task in build_tasks]

    def create_push_task(self, build_task: DockerPullOrBuildImageTask):
        parameter = dict(docker_image_task_module=build_task.__module__,
                         docker_image_task_class=build_task.__class__.__name__,
                         docker_image_task_params=build_task.param_kwargs,
                         force_push=self.force_push)
        return DockerPush_BuildTaskFromClassName(**parameter)
