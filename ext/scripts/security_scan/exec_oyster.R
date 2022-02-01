library("oysteR")
audit <- audit_installed_r_pkgs()

install.packages("knitr")

audit_vuln <- audit[audit$no_of_vulnerabilities > 0,]

library(jsonlite)
args = commandArgs(trailingOnly=TRUE)

n_vulnerabilities =  nrow(audit_vuln)

report <- toJSON(audit_vuln)
write(report, paste0(args[1], "/oyster.json"))

exit_code <- 0
if(n_vulnerabilities > 0) {
    exit_code <- 1
    library(knitr)
    write(kable(audit_vuln), paste0(args[1], "/oyster.md"))
    kable(audit_vuln, format="simple")
    print(paste(n_vulnerabilities, "vulnerabilities found!"))
} else {
    write("No vulnerabilities found!", paste0(args[1], "/oyster.md"))
}

quit(status=exit_code)
