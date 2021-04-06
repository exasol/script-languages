import sys
import pandas as pd
package_list_1_path = sys.argv[1]
package_list_2_path = sys.argv[2]
package_list_1 = pd.read_csv(package_list_1_path, delimiter="|", names=["package","version"])
package_list_2 = pd.read_csv(package_list_2_path, delimiter="|", names=["package","version"])
diff = pd.merge(package_list_1, package_list_2, how='outer', on='package',sort=False)
diff = diff.sort_values("package")
diff["Updated"] = diff["version_x"]!=diff["version_y"]
diff = diff.reset_index(drop=True)
diff.to_markdown(sys.stdout)
print()
