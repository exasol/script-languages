def _java_local_repository_impl(repository_ctx):
    prefix = "/usr/lib/jvm/java-11-openjdk-amd64"
    if 'JAVA_PREFIX' in repository_ctx.os.environ:
        prefix = repository_ctx.os.environ['JAVA_PREFIX']
    print("java prefix in environment specified; %s"%prefix)

    defines = '"ENABLE_JAVA_VM"'

    build_file_content = """
cc_library(
    name = "{name}",
    srcs = ["{prefix}/lib/server/libjvm.so"],
    hdrs = glob(["{prefix}/include/*.h","{prefix}/include/linux/*.h"]),
    includes = ["{prefix}/include","{prefix}/include/linux"],
    defines = [{defines}],
    copts = ["-ljvm"],
    visibility = ["//visibility:public"]
)""".format( name=repository_ctx.name, defines=defines, prefix="java")
    print(build_file_content)

    repository_ctx.symlink(prefix, "./java")
    repository_ctx.file("BUILD", build_file_content)

java_local_repository = repository_rule(
    implementation=_java_local_repository_impl,
    local = True,
    environ = ["JAVA_PREFIX"])

