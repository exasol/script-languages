from build_utils.docker_build import *
from build_utils.lib.docker.docker_push_task import DockerPushImageTask

class DockerPush_UDFClientDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_UDFClientDeps(flavor_path=flavor_path)


class DockerPush_LanguageDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_LanguageDeps(flavor_path=flavor_path)


class DockerPush_BuildDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_BuildDeps(flavor_path=flavor_path)


class DockerPush_BuildRun(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_BuildRun(flavor_path=flavor_path)


class DockerPush_BaseTestDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_BaseTestDeps(flavor_path=flavor_path)


class DockerPush_BaseTestBuildRun(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_BaseTestBuildRun(flavor_path=flavor_path)

class DockerPush_FlavorBaseDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_FlavorBaseDeps(flavor_path=flavor_path)


class DockerPush_FlavorCustomization(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_FlavorCustomization(flavor_path=flavor_path)


class DockerPush_FlavorTestBuildRun(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_FlavorTestBuildRun(flavor_path=flavor_path)


class DockerPush_Release(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_docker_image_task(self, flavor_path):
        return DockerBuild_Release(flavor_path=flavor_path)


class DockerPush(FlavorWrapperTask):

    def requires(self):
        return [self.generate_tasks_for_flavor(flavor_path) for flavor_path in self.actual_flavor_paths]

    def generate_tasks_for_flavor(self, flavor_path):
        return [DockerPush_UDFClientDeps(flavor_path=flavor_path),
                DockerPush_LanguageDeps(flavor_path=flavor_path),
                DockerPush_BuildDeps(flavor_path=flavor_path),
                DockerPush_BuildRun(flavor_path=flavor_path),
                DockerPush_BaseTestDeps(flavor_path=flavor_path),
                DockerPush_BaseTestBuildRun(flavor_path=flavor_path),
                DockerPush_FlavorBaseDeps(flavor_path=flavor_path)]
