package trivypackage trivy  GNU nano 6.2 /kernel.rego                                                                             *
package trivy

import data.lib.trivy

default ignore = false

ignore {
        input.PkgName == "linux-libc-dev"
        regex.match("^kernel:", input.Title)
}