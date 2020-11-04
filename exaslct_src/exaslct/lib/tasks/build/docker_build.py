import luigi
from exasol_integration_test_docker_environment.lib.base.flavor_task import FlavorsBaseTask
from luigi import Config

from exaslct_src.exaslct.lib.tasks.build.docker_flavor_build_base import DockerFlavorBuildBase


class DockerBuildParameter(Config):
    goals = luigi.ListParameter()
    shortcut_build = luigi.BoolParameter(True)


class DockerBuild(FlavorsBaseTask, DockerBuildParameter):
    def register_required(self):
        tasks = self.create_tasks_for_flavors_with_common_params(DockerFlavorBuild)
        self.register_dependencies(tasks)

    def run_task(self):
        pass


class DockerFlavorBuild(DockerFlavorBuildBase, DockerBuildParameter):

    def get_goals(self):
        return self.goals

    def run_task(self):
        build_tasks = self.create_build_tasks(self.shortcut_build)
        yield from self.run_dependencies(build_tasks)
