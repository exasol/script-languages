import json
from typing import Dict

import luigi

from build_utils.build_config import build_config
from build_utils.docker_pull_or_build_flavor_image_task import DockerPullOrBuildFlavorImageTask
from build_utils.image_dependency_collector import ImageDependencyCollector
from build_utils.image_info import ImageInfo

tasks = {}


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

    def requires(self):
        return {"language_deps": DockerBuild_LanguageDeps()}


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


class DockerBuild(luigi.Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/docker_build"
            % (self._build_config.ouput_directory))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return self._target

    def requires(self):
        return {
            # "build_run": DockerBuild_BuildRun(),
            #     "flavor_base_deps": DockerBuild_FlavorBaseDeps(),
            #     "base_test_build_run": DockerBuild_BaseTestBuildRun(),
            #     "flavor_test_build_run": DockerBuild_FlavorTestBuildRun(),
            "release": DockerBuild_Release()
        }

    def run(self):
        image_info_of_dependencies = ImageDependencyCollector().get_image_info_of_dependencies(self.input())
        with self.output().open("w") as file:
            json.dump(ImageInfo.merge_dependencies(image_info_of_dependencies), file)
