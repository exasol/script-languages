
def _ssl_local_repository_impl(repository_ctx):
    build_file_content = """
cc_library(
    name = "ssl",
    srcs = ["usr/lib/x86_64-linux-gnu/libssl.so","usr/lib/x86_64-linux-gnu/libcrypto.so"],
    hdrs = glob(["usr/include/openssl/*.h"]),
    includes = ["usr/include/openssl"],
    visibility = ["//visibility:public"]
 )
    """
    print(build_file_content)
    repository_ctx.file("BUILD", build_file_content)

ssl_local_repository = repository_rule(
    implementation=_ssl_local_repository_impl,
    local = True)

