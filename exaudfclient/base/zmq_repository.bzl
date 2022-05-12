def _zmq_local_repository_impl(repository_ctx):
    if 'ZMQ_LIBRARY_PREFIX' in repository_ctx.os.environ:
        library_prefix = repository_ctx.os.environ['ZMQ_LIBRARY_PREFIX']
    else:
        fail("Environment Variable ZMQ_LIBRARY_PREFIX not found")
    print("zmq library prefix in environment specified; %s"%library_prefix)

    if 'ZMQ_INCLUDE_PREFIX' in repository_ctx.os.environ:
        include_prefix = repository_ctx.os.environ['ZMQ_INCLUDE_PREFIX']
    else:
        fail("Environment Variable ZMQ_INCLUDE_PREFIX not found")
    print("zmq include prefix in environment specified; %s"%include_prefix)
    build_file_content = """
cc_library(
    name = "{name}",
    srcs = glob(["zmq/lib/**/libzmq.so"]),
    hdrs = glob(["zmq/include/zmq*"]),
    includes = ["zmq/include/"],
    visibility = ["//visibility:public"]
)""".format( name=repository_ctx.name)
    print(build_file_content)

    repository_ctx.symlink(library_prefix, "./zmq/lib")
    repository_ctx.symlink(include_prefix, "./zmq/include")
    repository_ctx.file("BUILD", build_file_content)

zmq_local_repository = repository_rule(
    implementation=_zmq_local_repository_impl,
    local = True,
    environ = ["ZMQ_LIBRARY_PREFIX","ZMQ_INCLUDE_PREFIX"])

