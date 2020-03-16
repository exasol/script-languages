Rscript -e 'installed.packages()[,c("Package","Version")]' | sed "s/  */|/g" | cut -f 2,3 -d "," | sed "s/\"//g" | sort -f -d
