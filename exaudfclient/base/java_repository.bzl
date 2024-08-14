load("@bazel_skylib//lib:paths.bzl", "paths")

# Workaround for the problems of JNI/JVM with rpath's used by Bazel.
# Problem Description:
# - Bazel creates for all external local repositories symlinks into its build directory
# - Bazel than writes during compilation these paths to the rpath of the binary
# - JNI/JVM uses the rpath for loading additonal shared libraries, 
#   but for a unknown reason it seems to truncate these pathes. 
# - We assume two possible reason, the symlink paths of bazel contain a @, or
#   the paths are to long and get for that reason truncated
# - In the end, it we wasn't able to convinve JNI/JVM from other pathes 
#   for its libraries, as the original pathes where apt installed the files.
# Solution:
# - We solved the problem by adding the original pahtes for the libraries into the rpath via the linkopts

def _find_shared_libraries(prefix,library,p_repository_ctx):
    command_result = p_repository_ctx.execute(["find", '%s'%prefix,'-name','%s'%library]) #TODO only one result
    if command_result.return_code != 0:
        fail("Could not acquire path of libjvm.so, got return code %s stderr: \n %s"
            % (command_result.return_code, command_result.stderr))
    path_to_library = command_result.stdout.strip("\n")
    print("path to %s: %s"%(library,path_to_library))
    return path_to_library

def _java_local_repository_impl(repository_ctx):
    prefix = "/usr/lib/jvm/java-11-openjdk-amd64"
    if 'JAVA_PREFIX' in repository_ctx.os.environ:
        prefix = repository_ctx.os.environ['JAVA_PREFIX']
    print("java prefix in environment specified; %s"%prefix)

    path_to_libjvm = paths.dirname(_find_shared_libraries(prefix,"libjvm.so",repository_ctx))

    defines = '"ENABLE_JAVA_VM"'
    build_file_content = """
cc_library(
    name = "java",
    srcs = glob(["{prefix}/include/*.h"], allow_empty=False),
    hdrs = glob(["{prefix}/include/*.h","{prefix}/include/linux/*.h"], allow_empty=False),
    includes = ["{prefix}/include","{prefix}/include/linux"],
    defines = [{defines}],
    linkopts = ["-ljvm","-L{rpath_libjvm}",'-Wl,-rpath','{rpath_libjvm}'],
    visibility = ["//visibility:public"]
)""".format( defines=defines, prefix="java", rpath_libjvm=path_to_libjvm)
    print(build_file_content)

    repository_ctx.symlink(prefix, "./java")
    repository_ctx.file("BUILD", build_file_content)

java_local_repository = repository_rule(
    implementation=_java_local_repository_impl,
    local = True,
    environ = ["JAVA_PREFIX"])

