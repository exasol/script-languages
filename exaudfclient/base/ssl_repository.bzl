def _ssl_local_repository_impl(repository_ctx):
    if 'OPENSSL_LIBRARY_PREFIX' in repository_ctx.os.environ:
        library_prefix = repository_ctx.os.environ['OPENSSL_LIBRARY_PREFIX']
    else:
        fail("Environment Variable OPENSSL_LIBRARY_PREFIX not found")
    print("openssl library prefix in environment specified; %s"%library_prefix)

    if 'OPENSSL_INCLUDE_PREFIX' in repository_ctx.os.environ:
        include_prefix = repository_ctx.os.environ['OPENSSL_INCLUDE_PREFIX']
    else:
        fail("Environment Variable OPENSSL_INCLUDE_PREFIX not found")
    print("openssl include prefix in environment specified; %s"%include_prefix)
    build_file_content = """
cc_library(
    name = "ssl",
    srcs = ["openssl/lib/libcrypto.so", ],
    hdrs = glob(["openssl/include/**"]),
    includes = ["openssl/include/"],
    visibility = ["//visibility:public"]
)"""
    print(build_file_content)

    repository_ctx.symlink(library_prefix+"/libcrypto.so", "./openssl/lib/libcrypto.so")
    repository_ctx.symlink(library_prefix+"/libssl.so", "./openssl/lib/libssl.so")
    repository_ctx.symlink(include_prefix, "./openssl/include")
    repository_ctx.file("BUILD", build_file_content)

ssl_local_repository = repository_rule(
    implementation=_ssl_local_repository_impl,
    local = True,
    environ = ["OPENSSL_LIBRARY_PREFIX","OPENSSL_LIBRARY_PREFIX"])

