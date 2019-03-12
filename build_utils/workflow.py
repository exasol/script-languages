from typing import Dict

import luigi

from build_utils.docker_pull_or_build_flavor_image_task import DockerPullOrBuildFlavorImageTask

tasks = {}


class DockerBuild_UDFClientDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "udfclient_deps"

    def get_additional_build_directories_mapping(self)->Dict[str,str]:
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

    def get_additional_build_directories_mapping(self)->Dict[str,str]:
        return {"src": "src"}

class DockerBuild_FlavorBaseDeps(DockerPullOrBuildFlavorImageTask):

    def get_build_step(self) -> str:
        return "flavor_base_deps"

    def requires(self):
        return {"language_deps": DockerBuild_LanguageDeps()}

class DockerBuild(luigi.WrapperTask):

    def requires(self):
        return [DockerBuild_UDFClientDeps(), DockerBuild_LanguageDeps(),
                DockerBuild_BuildDeps(), DockerBuild_BuildRun(),
                DockerBuild_FlavorBaseDeps()]

