BUILD_STEP=udfclient_deps
BUILD_STEP_PATH=flavor_base
bash generate_package_diff.sh ../../flavors/standard-EXASOL-7.0.0/$BUILD_STEP_PATH/$BUILD_STEP/packages/apt_get_packages ../../flavors/standard-EXASOL-7.1.0/$BUILD_STEP_PATH/$BUILD_STEP/packages/apt_get_packages  > diff_${BUILD_STEP}_apt_get_packages.md
BUILD_STEP=language_deps
BUILD_STEP_PATH=flavor_base
bash generate_package_diff.sh ../../flavors/standard-EXASOL-7.0.0/$BUILD_STEP_PATH/$BUILD_STEP/packages/apt_get_packages ../../flavors/standard-EXASOL-7.1.0/$BUILD_STEP_PATH/$BUILD_STEP/packages/apt_get_packages  > diff_${BUILD_STEP}_apt_get_packages.md
BUILD_STEP=flavor_base_deps
BUILD_STEP_PATH=flavor_base
bash generate_package_diff.sh ../../flavors/standard-EXASOL-7.0.0/$BUILD_STEP_PATH/$BUILD_STEP/packages/apt_get_packages ../../flavors/standard-EXASOL-7.1.0/$BUILD_STEP_PATH/$BUILD_STEP/packages/apt_get_packages  > diff_${BUILD_STEP}_apt_get_packages.md
BUILD_STEP=flavor_customization
BUILD_STEP_PATH=
bash generate_package_diff.sh ../../flavors/standard-EXASOL-7.0.0/$BUILD_STEP_PATH/$BUILD_STEP/packages/apt_get_packages ../../flavors/standard-EXASOL-7.1.0/$BUILD_STEP_PATH/$BUILD_STEP/packages/apt_get_packages  > diff_${BUILD_STEP}_apt_get_packages.md
