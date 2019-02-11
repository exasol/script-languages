def _get_actual_python_version(binary,p_repository_ctx):
    command_result = p_repository_ctx.execute([binary, "-c", """import sys; print(".".join(map(str, sys.version_info[:3])))"""])
    if command_result.return_code != 0:
        fail("Could not acquire actual python version, got return code %s stderr: \n %s"
            % (command_result.return_code, command_result.stderr))
    actual_version = command_result.stdout.strip("\n")
    print("python actual_version: %s"%actual_version)
    return actual_version

def _get_includes_and_hdrs(config_binary,p_repository_ctx):
    command_result = p_repository_ctx.execute([config_binary,"--includes"]) #example stdout: -I/usr/include/python3.6m -I/usr/include/python3.6m\n
    if command_result.return_code != 0:
        fail("Could not acquire includes for python, got return code %s stderr: \n %s"
            % (command_result.return_code, command_result.stderr))
    raw_includes = command_result.stdout.strip("\n") #example: -I/usr/include/python3.6m -I/usr/include/python3.6m
    raw_include_prefix_removal_length = 2 + 1
    splitted_cleaned_includes =  [i[raw_include_prefix_removal_length:] for i in raw_includes.split(" ")] #example: ["usr/include/python3.6m","usr/include/python3.6m"]
    
    includes = ",".join(['"%s"'%i for i in splitted_cleaned_includes] ) #example: ["\"/usr/include/python3.6m\"","\"/usr/include/python3.6m\""]
    print("python includes: %s"%includes)

    hdrs = ",".join(['"%s/*.h"'%i for i in splitted_cleaned_includes] ) #example: ["\"/usr/include/python3.6m/*.h\"","\"/usr/include/python3.6m/*.h\""]
    print("python hdrs: %s"%hdrs)

    return includes, hdrs

def _get_config_dir(config_binary,repository_ctx):
    command_result = repository_ctx.execute([config_binary,"--configdir"])
    if command_result.return_code != 0:
        fail("Could not acquire config_dir for python, got return code %s stderr: \n %s"
            % (command_result.return_code, command_result.stderr))
    #print(command_result.stdout,command_result.stderr)
    config_dir = command_result.stdout.strip("\n")[1:]
    print("python config_dir: %s"%config_dir)
    return config_dir

def _python_local_repository_impl(repository_ctx):
    prefix = "/usr"
    if 'PYTHON_PREFIX' in repository_ctx.os.environ:
        prefix = repository_ctx.os.environ['PYTHON_PREFIX']
    print("python prefix in environment specified; %s"%prefix)

    version = "python2.7"
    if 'PYTHON_VERSION' in repository_ctx.os.environ:
        version = repository_ctx.os.environ['PYTHON_VERSION']
    print("python version in environment specified; %s"%version)

    binary = prefix+"/bin/"+version
    actual_version = _get_actual_python_version(binary,repository_ctx)
    
    config_binary = binary + "-config"
    includes, hdrs = _get_includes_and_hdrs(config_binary,repository_ctx)
    config_dir = _get_config_dir(config_binary,repository_ctx)

    defines='"ENABLE_PYTHON_VM"'
    if actual_version[0]=="3":
        defines = defines+',"ENABLE_PYTHON3"'

    build_file_content = """
cc_library(
    name = "{name}",
    srcs = glob(["{config_dir}/*.so"]),
    hdrs = glob([{hdrs}]),
    includes = [{includes}],
    defines = [{defines}],
    visibility = ["//visibility:public"]
)""".format(config_dir=config_dir, hdrs=hdrs, includes=includes, name=repository_ctx.name, defines=defines)
    print(build_file_content)

    repository_ctx.symlink(prefix, "."+prefix)
    repository_ctx.file("BUILD", build_file_content)

python_local_repository = repository_rule(
    implementation=_python_local_repository_impl,
    local = True,
    environ = ["PYTHON_PREFIX","PYTHON_VERSION"])


def _get_numpy_include_dir(binary,p_repository_ctx):
    command_result = p_repository_ctx.execute([binary, "-c", """import numpy as np; print(np.get_include())"""])
    if command_result.return_code != 0:
        fail("Could not acquire numpy include dir, got return code %s stderr: \n %s"
            % (command_result.return_code, command_result.stderr))
    numpy_include_dir = command_result.stdout.strip("\n")[1:]
    print("python numpy_include_dir: %s"%numpy_include_dir)
    return numpy_include_dir


def _numpy_local_repository_impl(repository_ctx):
    prefix = "/usr"
    if 'PYTHON_PREFIX' in repository_ctx.os.environ:
        prefix = repository_ctx.os.environ['PYTHON_PREFIX']
    print("python prefix in environment specified; %s"%prefix)

    version = "python2.7"
    if 'PYTHON_VERSION' in repository_ctx.os.environ:
        version = repository_ctx.os.environ['PYTHON_VERSION']
    print("python version in environment specified; %s"%version)

    binary = prefix+"/bin/"+version
    actual_version = _get_actual_python_version(binary,repository_ctx)

    defines='"ENABLE_PYTHON_VM"'
    if actual_version[0]=="3":
        defines = defines+',"ENABLE_PYTHON3"'

    numpy_include_dir =_get_numpy_include_dir(binary,repository_ctx)
    hdrs = numpy_include_dir + "/*/*.h"
    build_file_content = """
cc_library(
    name = "{name}",
    srcs = [],
    hdrs = glob(["{hdrs}"]),
    includes = ["{includes}"],
    defines = [{defines}],
    visibility = ["//visibility:public"]
)""".format(hdrs=hdrs, includes=numpy_include_dir, name=repository_ctx.name, defines=defines)
    print(build_file_content)

    repository_ctx.symlink(prefix, "."+prefix)
    repository_ctx.file("BUILD", build_file_content)

numpy_local_repository = repository_rule(
    implementation=_numpy_local_repository_impl,
    local = True,
    environ = ["PYTHON_PREFIX","PYTHON_VERSION"])

