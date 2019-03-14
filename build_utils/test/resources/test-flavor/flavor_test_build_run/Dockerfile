FROM {{flavor_customization}}

COPY --from={{base_test_build_run}} /exasol_emulator /exasol_emulator
COPY --from={{base_test_build_run}} /exaudf /exaudf
COPY --from={{base_test_build_run}} /exaudf_src /exaudf_src
COPY --from={{base_test_build_run}} /root/.cache/bazel /root/.cache/bazel