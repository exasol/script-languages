is_not_in_ignore_list <- function(vulnerabilities, ignore_list) {
  for (vulner in vulnerabilities) {
    for (vuln in vulner) {
      for (vul in vuln) {
        for (vul_id in vul$id) {
          if (!vul_id %in% ignore_list) {
            return(TRUE)
          }
        }
      }
    }
  }
  FALSE
}

filter_for_ignore_list <- function(df, ignore_list) {
  res <- data.frame()
  for (row in nrow(df)) {
    if (is_not_in_ignore_list(df[row,"vulnerabilities"], ignore_list)) {
      res <- rbind(res, df[row,])
    }
  }
  res
}

read_ignore_list <- function() {
  oyster_ignore_list_file <- "/.oysterignore"
  res <- c()
  if (file.exists(oyster_ignore_list_file)) {
    res <- read.table(oyster_ignore_list_file, header=FALSE)
  }
  res
}

filter_out_ignored_cves <- function(df) {
 ignore_list <- read_ignore_list()
 filter_for_ignore_list(df, ignore_list)
}

library("oysteR")
audit <- audit_installed_r_pkgs()

install.packages("knitr")

audit_vuln <- audit[audit$no_of_vulnerabilities > 0,]

filtered_vuln <- filter_out_ignored_cves(audit_vuln)

library(jsonlite)
args = commandArgs(trailingOnly=TRUE)

n_vulnerabilities =  nrow(filtered_vuln)

report <- toJSON(filtered_vuln)
write(report, paste0(args[1], "/oyster.json"))

exit_code <- 0
if(n_vulnerabilities > 0) {
    exit_code <- 1
    library(knitr)
    write(kable(filtered_vuln), paste0(args[1], "/oyster.md"))
    print(kable(filtered_vuln, format="simple"))
    print(paste(n_vulnerabilities, "vulnerabilities found!"))
} else {
    write("No vulnerabilities found!", paste0(args[1], "/oyster.md"))
}

quit(status=exit_code)
