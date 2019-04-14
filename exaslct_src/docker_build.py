from typing import Dict

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.docker.docker_pull_or_build_flavor_image_task import DockerPullOrBuildFlavorImageTask
from exaslct_src.lib.flavor_task import FlavorWrapperTask


class DockerBuild_UDFClientDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "udfclient_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"01_nodoc": "ext/01_nodoc"}


class DockerBuild_LanguageDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "language_deps"

    def requires(self):
        return {"udfclient_deps": DockerBuild_UDFClientDeps(flavor_path=self.flavor_path)}


class DockerBuild_BuildDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "build_deps"

    def requires(self):
        return {"language_deps": DockerBuild_LanguageDeps(flavor_path=self.flavor_path)}


class DockerBuild_BuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "build_run"

    def requires(self):
        return {"build_deps": DockerBuild_BuildDeps(flavor_path=self.flavor_path)}

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"src": "src"}


class DockerBuild_BaseTestDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "base_test_deps"

    def requires(self):
        return {"build_deps": DockerBuild_BuildDeps(flavor_path=self.flavor_path)}


class DockerBuild_BaseTestBuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "base_test_build_run"

    def requires(self):
        return {"base_test_deps": DockerBuild_BaseTestDeps(flavor_path=self.flavor_path)}

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"src": "src", "emulator": "emulator"}


class DockerBuild_FlavorBaseDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_base_deps"

    def get_additional_build_directories_mapping(self):
        return {"01_nodoc": "ext/01_nodoc"}


class DockerBuild_FlavorCustomization(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_customization"

    def requires(self):
        return {"flavor_base_deps": DockerBuild_FlavorBaseDeps(flavor_path=self.flavor_path)}


class DockerBuild_FlavorTestBuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_test_build_run"

    def requires(self):
        return {"flavor_customization": DockerBuild_FlavorCustomization(flavor_path=self.flavor_path),
                "base_test_build_run": DockerBuild_BaseTestBuildRun(flavor_path=self.flavor_path)}


class DockerBuild_Release(DockerPullOrBuildFlavorImageTask):
    def get_build_step(self) -> str:
        return "release"

    def requires(self):
        return {"flavor_customization": DockerBuild_FlavorCustomization(flavor_path=self.flavor_path),
                "build_run": DockerBuild_BuildRun(flavor_path=self.flavor_path),
                "language_deps": DockerBuild_LanguageDeps(flavor_path=self.flavor_path)}


class DockerBuild(FlavorWrapperTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()

    def requires(self):
        return [self.generate_tasks_for_flavor(flavor_path) for flavor_path in self.actual_flavor_paths]

    def generate_tasks_for_flavor(self, flavor_path):
        return [DockerBuild_BaseTestBuildRun(flavor_path=flavor_path),
                DockerBuild_FlavorTestBuildRun(flavor_path=flavor_path),
                DockerBuild_Release(flavor_path=flavor_path)]
