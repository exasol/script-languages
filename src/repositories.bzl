def _python27_local_repository_impl(repository_ctx):
    if 'PYTHON_PREFIX' in repository_ctx.os.environ:
        path = repository_ctx.os.environ['PYTHON_PREFIX']
        if path == "":
            path = "/usr"
    else:
        path = "/usr"
    repository_ctx.symlink(path, "usr")
    repository_ctx.file("BUILD", """
cc_library(
    name = "test",
    srcs = ["usr/lib/python2.7/config-x86_64-linux-gnu/libpython2.7.so"],
    hdrs = glob(["usr/include/python2.7/*.h"]),
    includes = ["usr/include/python2.7"],
    visibility = ["//visibility:public"]
)
  """)

python27_local_repository = repository_rule(
    
    implementation=_python27_local_repository_impl,
    local = True,
    environ = [])

