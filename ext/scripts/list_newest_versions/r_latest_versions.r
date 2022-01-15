#!/usr/bin/env Rscript
library(dplyr)
library(RWsearch)

args = commandArgs(trailingOnly=TRUE)
if (length(args)!=2) {
  stop("Usage: ./r_latest_versions.r <input_package_list_file> <output_package_list_file>  .n", call.=FALSE)
}
input_file = args[1]
output_file = args[2]

input = read.table(file = input_file, sep = "|", comment.char = "#")
packages = input[,1]

#archivedb_down(filename = "CRAN-archive.html", dir = tempdir(), url = "https://cran.r-project.org/src/contrib/Archive")
#packages=c("RCurl","tidyverse")
#archive_version = p_archive_lst(packages)
#previous_versions = archive_version[["RCurl"]][["Name"]]


crandb_down(dir = tempdir(), repos = "https://cloud.r-project.org")
df = crandb[,c("Package","Version")]
latest_versions = df %>% filter(Package %in% packages)
write.table(latest_versions, file = output_file, quote = FALSE, sep="|", col.names=FALSE, row.names=FALSE)
