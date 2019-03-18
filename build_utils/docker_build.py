from typing import Dict

import luigi

from build_utils.docker_pull_or_build_flavor_image_task import DockerPullOrBuildFlavorImageTask

class DockerBuild_UDFClientDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "udfclient_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"ext": "ext"}


class DockerBuild_LanguageDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "language_deps"

    def requires(self):
        return {"udfclient_deps": DockerBuild_UDFClientDeps()}


class DockerBuild_BuildDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "build_deps"

    def requires(self):
        return {"language_deps": DockerBuild_LanguageDeps()}


class DockerBuild_BuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "build_run"

    def requires(self):
        return {"build_deps": DockerBuild_BuildDeps()}

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"src": "src"}


class DockerBuild_BaseTestDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "base_test_deps"

    def requires(self):
        return {"build_deps": DockerBuild_BuildDeps()}


class DockerBuild_BaseTestBuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "base_test_build_run"

    def requires(self):
        return {"base_test_deps": DockerBuild_BaseTestDeps()}

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"src": "src", "emulator": "emulator"}


class DockerBuild_FlavorBaseDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_base_deps"

    def get_additional_build_directories_mapping(self):
        return {"ext": "ext"}

class DockerBuild_FlavorCustomization(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_customization"

    def requires(self):
        return {"flavor_base_deps": DockerBuild_FlavorBaseDeps()}


class DockerBuild_FlavorTestBuildRun(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_test_build_run"

    def requires(self):
        return {"flavor_customization": DockerBuild_FlavorCustomization(),
                "base_test_build_run": DockerBuild_BaseTestBuildRun()}


class DockerBuild_Release(DockerPullOrBuildFlavorImageTask):
    def get_build_step(self) -> str:
        return "release"

    def requires(self):
        return {"flavor_customization": DockerBuild_FlavorCustomization(),
                "build_run": DockerBuild_BuildRun()}

class DockerBuild(luigi.WrapperTask):

    def requires(self):
        return [DockerBuild_UDFClientDeps(),
                DockerBuild_LanguageDeps(),
                DockerBuild_BuildDeps(),
                DockerBuild_BuildRun(),
                DockerBuild_BaseTestDeps(),
                DockerBuild_BaseTestBuildRun(),
                DockerBuild_FlavorBaseDeps(),
                DockerBuild_FlavorCustomization(),
                DockerBuild_FlavorTestBuildRun(),
                DockerBuild_Release()]