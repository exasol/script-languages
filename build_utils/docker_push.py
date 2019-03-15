import luigi

from build_utils.docker_build import *
from build_utils.docker_push_task import DockerPushImageTask

class DockerPush_UDFClientDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_UDFClientDeps()

class DockerPush_LanguageDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_LanguageDeps()

class DockerPush_BuildDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_BuildDeps()

class DockerPush_BuildRun(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_BuildRun()

class DockerPush_BaseTestDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_BaseTestDeps()

class DockerPush_BaseTestBuildRun(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_BaseTestBuildRun()

class DockerPush_FlavorBaseDeps(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_FlavorBaseDeps()

class DockerPush_FlavorCustomization(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_FlavorCustomization()


class DockerPush_FlavorTestBuildRun(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_FlavorTestBuildRun()

class DockerPush_Release(DockerPushImageTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return DockerBuild_Release()

class DockerPush(luigi.WrapperTask):

    def requires(self):
        return [DockerPush_UDFClientDeps(),
                DockerPush_LanguageDeps(),
                DockerPush_BuildDeps(),
                DockerPush_BuildRun(),
                DockerPush_BaseTestDeps(),
                DockerPush_BaseTestBuildRun(),
                DockerPush_FlavorBaseDeps()]