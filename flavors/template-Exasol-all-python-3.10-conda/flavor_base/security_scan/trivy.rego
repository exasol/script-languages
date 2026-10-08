package trivy

import data.lib.trivy

default ignore = false

ignore {
        input.PkgName == "linux-libc-dev"
        regex.match("^([kK]ernel:|In the Linux kernel)", input.Title)
}
