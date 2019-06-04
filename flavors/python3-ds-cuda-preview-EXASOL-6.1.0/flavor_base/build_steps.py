from typing import Dict

from exaslct_src.lib.docker.docker_flavor_image_task import DockerFlavorAnalyzeImageTask


class AnalyzeCudaDeps(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "cuda_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"01_nodoc": "ext/01_nodoc"}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeUDFClientDeps(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "udfclient_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"01_nodoc": "ext/01_nodoc"}

    def requires_tasks(self):
        return {"cuda_deps": AnalyzeCudaDeps(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeLanguageDeps(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "language_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"scripts": "ext/scripts"}

    def requires_tasks(self):
        return {"udfclient_deps": AnalyzeUDFClientDeps(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeBuildDeps(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "build_deps"

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"01_nodoc": "ext/01_nodoc", "scripts": "ext/scripts"}

    def requires_tasks(self):
        return {"cuda_deps": AnalyzeCudaDeps(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeBuildRun(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "build_run"

    def requires_tasks(self):
        return {"build_deps": AnalyzeBuildDeps(flavor_path=self.flavor_path),
                "language_deps": AnalyzeLanguageDeps(flavor_path=self.flavor_path)}

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"src": "src"}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeBaseTestDeps(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "base_test_deps"

    def requires_tasks(self):
        return {"build_deps": AnalyzeBuildDeps(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeBaseTestBuildRun(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "base_test_build_run"

    def requires_tasks(self):
        return {"base_test_deps": AnalyzeBaseTestDeps(flavor_path=self.flavor_path),
                "language_deps": AnalyzeLanguageDeps(flavor_path=self.flavor_path)}

    def get_additional_build_directories_mapping(self) -> Dict[str, str]:
        return {"src": "src", "emulator": "emulator"}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeFlavorBaseDeps(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "flavor_base_deps"

    def requires_tasks(self):
        return {"language_deps": AnalyzeLanguageDeps(flavor_path=self.flavor_path)}

    def get_additional_build_directories_mapping(self):
        return {"01_nodoc": "ext/01_nodoc", "scripts": "ext/scripts"}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeFlavorCustomization(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "flavor_customization"

    def requires_tasks(self):
        return {"flavor_base_deps": AnalyzeFlavorBaseDeps(flavor_path=self.flavor_path)}


class AnalyzeFlavorTestBuildRun(DockerFlavorAnalyzeImageTask):

    def get_build_step(self) -> str:
        return "flavor_test_build_run"

    def requires_tasks(self):
        return {"flavor_customization": AnalyzeFlavorCustomization(flavor_path=self.flavor_path),
                "base_test_build_run": AnalyzeBaseTestBuildRun(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"


class AnalyzeRelease(DockerFlavorAnalyzeImageTask):
    def get_build_step(self) -> str:
        return "release"

    def requires_tasks(self):
        return {
            "cuda_deps": AnalyzeCudaDeps(flavor_path=self.flavor_path),
            "flavor_customization": AnalyzeFlavorCustomization(flavor_path=self.flavor_path),
            "build_run": AnalyzeBuildRun(flavor_path=self.flavor_path),
            "language_deps": AnalyzeLanguageDeps(flavor_path=self.flavor_path)}

    def get_path_in_flavor(self):
        return "flavor_base"
