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

def _find_shared_libraries(prefix,p_repository_ctx):
    command_result = p_repository_ctx.execute(["find", '%s'%prefix,'-name','*.so*',"-maxdepth","1"])
    if command_result.return_code != 0:
        fail("Error while searching for shared_libraries in %s, got return code %s stderr: \n %s"
            % (prefix,command_result.return_code, command_result.stderr))
    path_to_libraries = command_result.stdout.split("\n")
    print("found following libraries in %s: %s"%(prefix,path_to_libraries))
    return path_to_libraries

def _find_shared_library(prefix,library,p_repository_ctx):
    command_result = p_repository_ctx.execute(["find", '%s'%prefix,'-name','%s'%library]) #TODO only one result
    if command_result.return_code != 0:
        fail("Could not acquire path of %s, got return code %s stderr: \n %s"
            % (library,command_result.return_code, command_result.stderr))
    path_to_library = command_result.stdout.strip("\n")
    print("path to %s: %s"%(library,path_to_library))
    return path_to_library

def _build_linkopts(prefix,lib,p_repository_ctx):
    lib_filename="lib%s.so"%lib
    linkopts_for_lib = """"-l{lib}","-L{path_to_lib}",'-Wl,-rpath','{path_to_lib}'""".format(lib=lib,path_to_lib=prefix)
    print("linkopts_for_lib: %s"%linkopts_for_lib)
    return linkopts_for_lib

def _java_local_repository_impl(repository_ctx):
    prefix = "/usr/lib/jvm/java-11-openjdk-amd64"
    if 'JAVA_PREFIX' in repository_ctx.os.environ:
        prefix = repository_ctx.os.environ['JAVA_PREFIX']
    print("java prefix in environment specified; %s"%prefix)

    path_to_jdk_lib = prefix+"/lib"
    found_jdk_libs = _find_shared_libraries(path_to_jdk_lib,repository_ctx)
    jdk_lib_file_names = [paths.basename(lib_path) for lib_path in found_jdk_libs]
    jdk_lib_names = [lib_file_name[3:][:-3] for lib_file_name in jdk_lib_file_names 
		     if lib_file_name!=""
		     and lib_file_name!="libatk-wrapper.so"
		     ]
    print("jdk_lib_file_names: %s"%jdk_lib_file_names)
    print("jdk_lib_names: %s"%jdk_lib_names)
    libjvm_path = paths.dirname(_find_shared_library(prefix,"libjvm.so",repository_ctx))
    linkopts_for_lib = [_build_linkopts(libjvm_path,"jvm",repository_ctx)]
    for lib in jdk_lib_names:
        linkopts_for_lib += [_build_linkopts(path_to_jdk_lib,lib,repository_ctx)]
    defines = '"ENABLE_JAVA_VM"'
    build_file_content = """
cc_library(
    name = "{name}",
    srcs = glob(["{prefix}/include/*.h"], allow_empty=False),
    hdrs = glob(["{prefix}/include/*.h","{prefix}/include/linux/*.h"], allow_empty=False),
    includes = ["{prefix}/include","{prefix}/include/linux"],
    defines = [{defines}],
    linkopts = [{linkopts}],
    visibility = ["//visibility:public"]
)""".format( name=repository_ctx.name, defines=defines,prefix="java",linkopts=",".join(linkopts_for_lib) )
    print(build_file_content)

    repository_ctx.symlink(prefix, "./java")
    repository_ctx.file("BUILD", build_file_content)

java_local_repository = repository_rule(
    implementation=_java_local_repository_impl,
    local = True,
    environ = ["JAVA_PREFIX"])

