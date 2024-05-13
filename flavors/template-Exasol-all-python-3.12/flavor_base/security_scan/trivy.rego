package trivy

import data.lib.trivy

default ignore = false

ignore {
        input.PkgName == "linux-libc-dev"
        regex.match("^(kernel:|In the Linux kernel)", input.Title)
}