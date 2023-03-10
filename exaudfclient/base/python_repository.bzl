def _get_actual_python_version(binary,p_repository_ctx):
    command_result = p_repository_ctx.execute([binary, "-c", """import sys; print(".".join(map(str, sys.version_info[:3])))"""])
    if command_result.return_code != 0:
        fail("Could not acquire actual python version, got return code %s stderr: \n %s"
            % (command_result.return_code, command_result.stderr))
    actual_version = command_result.stdout.strip("\n")
    print("python actual_version: %s"%actual_version)
    return actual_version

def _get_sysconfig_value(binary,key,p_repository_ctx):
    script = "import sysconfig; print(sysconfig.get_config_var('{key}'))".format(key=key)
    command_result = p_repository_ctx.execute([binary,"-c", script])
    if command_result.return_code != 0:
        fail("Could not acquire {key} for python, got return code {return_code} stderr: \n {stderr}".format(
            key=key, return_code=command_result.return_code, stderr=command_result.stderr))
    stripped_command_result = command_result.stdout.strip("\n")
    return stripped_command_result

def _get_include_dir(binary,version,p_repository_ctx): 
    key = "INCLUDEDIR"
    base_include_dir = _get_sysconfig_value(binary,key,p_repository_ctx) #example: /usr/include
    include_dir = base_include_dir+"/"+version #example /usr/include/python3.8 
    print("python {key}: {include_dir}".format(key=key, include_dir=include_dir))
    return include_dir

def _get_lib_glob(binary, version, p_repository_ctx):
    key = "LIBDIR"
    base_lib_dir = _get_sysconfig_value(binary,key,p_repository_ctx) #example: /usr/lib
    lib_glob = "%s/**/lib%s*.so" % (base_lib_dir,version)
    print("python {key}_glob: {lib_glob}".format(key=key, lib_glob=lib_glob))
    return lib_glob

def _python_local_repository_impl(repository_ctx):
    python_prefix_env_var = repository_ctx.name.upper() + "_PREFIX"
    if python_prefix_env_var in repository_ctx.os.environ:
        prefix = repository_ctx.os.environ[python_prefix_env_var]
    else:
        fail("Environment Variable %s not found"%python_prefix_env_var)
    print("python prefix in environment specified; %s"%prefix)

    python_version_env_var = repository_ctx.name.upper() + "_VERSION"
    if python_version_env_var in repository_ctx.os.environ:
        version = repository_ctx.os.environ[python_version_env_var]
        if repository_ctx.name == "python3" and version.startswith("2"):
            fail("Wrong python version specified in environment variable %s, got binary name '%s', but version number '%s'"%(python_version_env_var,repository_ctx.name,version))
        if repository_ctx.name == "python" or version.startswith("2"):
            fail("Python 2 is not supported anymore, but specified in environment variable %s, got %s, %s"%(python_version_env_var,repository_ctx.name,version))
    else:
        fail("Environment Variable %s not found"%python_version_env_var)
    print("python version in environment specified; %s"%version)

    binary = prefix+"/bin/"+version
    actual_version = _get_actual_python_version(binary,repository_ctx)
    
    include_dir = _get_include_dir(binary, version, repository_ctx)
    lib_glob = _get_lib_glob(binary, version, repository_ctx)
    defines = ['"ENABLE_PYTHON_VM"']
    if actual_version[0]=="3":
        defines.append('"ENABLE_PYTHON3"')
    defines_str = ",".join(defines) 
    build_file_content = """
cc_library(
    name = "{name}",
    srcs = glob(["{lib_glob}"]),
    hdrs = glob(["{include_dir}/**/*.h"]),
    includes = ["{include_dir}"],
    defines = [{defines}],
    visibility = ["//visibility:public"]
)""".format(lib_glob=lib_glob[1:], include_dir=include_dir[1:], name=repository_ctx.name, defines=defines_str)
    print(build_file_content)

    repository_ctx.symlink(prefix, "."+prefix)
    repository_ctx.file("BUILD", build_file_content)

python_local_repository = repository_rule(
    implementation=_python_local_repository_impl,
    local = True,
    environ = ["PYTHON3_PREFIX", "PYTHON3_VERSION"])


def _get_numpy_include_dir(binary,p_repository_ctx):
    command_result = p_repository_ctx.execute([binary, "-c", """import numpy as np; print(np.get_include())"""])
    if command_result.return_code != 0:
        fail("Could not acquire numpy include dir, got return code %s stderr: \n %s"
            % (command_result.return_code, command_result.stderr))
    numpy_include_dir = command_result.stdout.strip("\n")[1:]
    print("python numpy_include_dir: %s"%numpy_include_dir)
    return numpy_include_dir


def _numpy_local_repository_impl(repository_ctx):
    python_prefix_env_var = "PYTHON3_PREFIX"
    if python_prefix_env_var in repository_ctx.os.environ:
        prefix = repository_ctx.os.environ[python_prefix_env_var]
    else:
        fail("Environment Variable %s not found"%python_prefix_env_var)
    print("python prefix in environment specified; %s"%prefix)

    python_version_env_var = "PYTHON3_VERSION"
    if python_version_env_var in repository_ctx.os.environ:
        version = repository_ctx.os.environ[python_version_env_var]
    else:
        fail("Environment Variable %s not found"%python_version_env_var)
    if version.startswith("2"):
        fail("Wrong python version specified in environment variable %s, got binary name '%s', but version number '%s'"%(python_version_env_var,repository_ctx.name,version))
    print("python version in environment specified; %s"%version)

    binary = prefix+"/bin/"+version
    actual_version = _get_actual_python_version(binary,repository_ctx)

    defines = ['"ENABLE_PYTHON_VM"']
    if actual_version[0]=="3":
        defines.append('"ENABLE_PYTHON3"')
    defines_str = ",".join(defines) 

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
)""".format(hdrs=hdrs, includes=numpy_include_dir, name=repository_ctx.name, defines=defines_str)
    print(build_file_content)

    repository_ctx.symlink(prefix, "."+prefix)
    repository_ctx.file("BUILD", build_file_content)

numpy_local_repository = repository_rule(
    implementation=_numpy_local_repository_impl,
    local = True,
    environ = ["PYTHON3_PREFIX","PYTHON3_VERSION"])

