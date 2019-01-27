bazel query --incompatible_remove_native_http_archive=false --incompatible_package_name_is_a_function=false "deps(//:$1) except deps(@org_pubref_rules_protobuf//cpp)" --output graph > graph.in
dot -Tpng < graph.in > graph.png
