from __future__ import annotations

from pathlib import Path

from exasol.slc_ci.model.build_config import BuildConfig

SLC_BUILD_CONFIG = BuildConfig(root=Path(__file__).parent, base_branch="origin/master",
                               ignore_paths=[
                                   ".gitignore",
                                   ".dockerignore",
                                   "LICENSE",
                                   "README.md",
                                   "find_duplicate_error_codes.sh",
                                   "find_error_codes.sh",
                                   "find_highest_error_codes_per_module.sh",
                                   "find_incomplete_error_codes.sh",
                                   "find_next_error_code_per_module.sh",
                                   "show_code_arround_error_codes.sh",
                                   "visualize_task_dependencies.sh",
                                   "udf-script-signature-generator",
                                   ".github",
                                   "emulator",
                                   "githooks"
                               ],
                               docker_build_repository="exadockerci4/script-languages-build-cache",
                               docker_release_repository="exasol/script-language-container",
                               test_container_folder="test_container", )
