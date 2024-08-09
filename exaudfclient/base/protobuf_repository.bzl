def _protobuf_local_repository_impl(repository_ctx):
    if 'PROTOBUF_LIBRARY_PREFIX' in repository_ctx.os.environ:
        library_prefix = repository_ctx.os.environ['PROTOBUF_LIBRARY_PREFIX']
    else:
        fail("Environment Variable PROTOBUF_LIBRARY_PREFIX not found")
    print("protobuf library prefix in environment specified; %s"%library_prefix)

    if 'PROTOBUF_INCLUDE_PREFIX' in repository_ctx.os.environ:
        include_prefix = repository_ctx.os.environ['PROTOBUF_INCLUDE_PREFIX']
    else:
        fail("Environment Variable PROTOBUF_INCLUDE_PREFIX not found")
    print("protobuf include prefix in environment specified; %s"%include_prefix)
    build_file_content = """
cc_library(
    name = "protobuf",
    srcs = glob(["protobuf/lib/**/libprotobuf*.so"]),
    hdrs = glob(["protobuf/include/**"]),
    includes = ["protobuf/include/"],
    visibility = ["//visibility:public"]
)"""
    print(build_file_content)

    repository_ctx.symlink(library_prefix, "./protobuf/lib")
    repository_ctx.symlink(include_prefix, "./protobuf/include")
    repository_ctx.file("BUILD", build_file_content)

protobuf_local_repository = repository_rule(
    implementation=_protobuf_local_repository_impl,
    local = True,
    environ = ["PROTOBUF_LIBRARY_PREFIX","PROTOBUF_BIN","PROTOBUF_INCLUDE_PREFIX"])

