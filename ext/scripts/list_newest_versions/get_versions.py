import json
from  urllib.request import urlopen
import sys
from distutils.version import LooseVersion

def versions(package_name, current_package_version):
    url = "https://pypi.org/pypi/%s/json" % (package_name,)
    data = json.load(urlopen(url))
    try:
        versions = list(data["releases"].keys())
        versions.sort(key=LooseVersion)
    except:
        versions = list(data["releases"].keys())
        versions.sort(key=lambda x: x.split("."))
    index=versions.index(current_package_version)
    return versions[index:]

try:
    package_info = sys.argv[1]
    package_name, current_package_version = package_info.split("|")
    print("\n".join([f"{package_name}|{current_package_version}|{version}" for version in versions(package_name, current_package_version)]))
except Exception as e:
    print(f"Error looking up {package_name}: {e}",file=sys.stderr)

