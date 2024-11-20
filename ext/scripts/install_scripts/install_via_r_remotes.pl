#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS
  Installs R packages with the remotes package https://github.com/r-lib/remotes

  install_via_r_versions.pl [OPTIONS]
  Options:
    --help                Brief help message
    --dry-run             Doesn't execute the command, only prints it to STDOUT
    --file                Input file with each line represents a input. 
                          A line can have multiple elements separated by --element-separator. 
                          Lines everything after a # is interpreted as comment
    --with-versions       Uses versions specified in the input file in the second element of each line
    --allow-no-versions   If --with-versions is active, allow packages to have no version specified
    --rscript-binary      Rscript binary to use for installation
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use package_mgmt_utils;
use Getopt::Long;
use Pod::Usage;
#use IPC::System::Simple;

my $help = 0;
my $dry_run = 0;
my $file = '';
my $element_separator = "\\|";
my $rscript_binary = '';
my $with_versions = 0;
my $allow_no_version = 0;

GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "file=s" => \$file,
            "with-versions" => \$with_versions,
            "allow-no-version" => \$allow_no_version,
            "rscript-binary=s" => \$rscript_binary,
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
}

if($rscript_binary eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --rscript-binary was not specified",1);
}


my $combining_template = '
library(remotes)
install_or_fail <- function(package_name, version){

   tryCatch({install_version(package_name, version, repos="https://cloud.r-project.org", Ncpus=4)
         library(package_name, character.only = TRUE)},
         error = function(e){
             print(e)
             stop(paste("installation failed for:",package_name ))},
         warning = function(w){
           catch <-
             grepl("download of package .* failed", w$message) ||
             grepl("(dependenc|package).*(is|are) not available", w$message) ||
             grepl("installation of package.*had non-zero exit status", w$message) ||
             grepl("installation of one or more packages failed", w$message)
           if(catch){ print(w$message)
             stop(paste("installation failed for:",package_name ))}}
         )

 }

<<<<0>>>>
';

my $combining_template_validation = '

available_packages <- available.packages()

validate_or_fail <- function(package_name){
    # Check if the package is in the list of available packages
    is_installed <- package_name %in% rownames(available_packages)

    # Check the result
    if (!is_installed) {
        stop(paste("Package nor installed:", package_name))
    }
}


<<<<0>>>>
';


my @separators = ("\n");
my @templates = ('install_or_fail("<<<<0>>>>",NULL)');
if($with_versions){  
    @templates = ('install_or_fail("<<<<0>>>>","<<<<1>>>>")');
}

my @validation_templates = ('validate_or_fail("<<<<0>>>>")');

sub identity {
    my ($line) = @_;
    return $line 
}


sub replace_missing_version{
    my ($line) = @_;
    $line =~ s/"<<<<1>>>>"/NULL/g;
    return $line;
}

my @rendered_line_transformation_functions = (\&identity);
if($with_versions and $allow_no_version){
    @rendered_line_transformation_functions = (\&replace_missing_version);
}
my @rendered_line_transformation_functions_validation = (\&identity);

my $script = 
    package_mgmt_utils::generate_joined_and_transformed_string_from_file(
        $file,$element_separator,$combining_template,\@templates,\@separators,\@rendered_line_transformation_functions) .
    package_mgmt_utils::generate_joined_and_transformed_string_from_file(
        $file,$element_separator,$combining_template_validation,\@validation_templates,\@separators,\@rendered_line_transformation_functions_validation);



if($with_versions and not $allow_no_version){
    if (index($script, "<<<<1>>>>") != -1) {
        die "Command '$script' contains packages with unspecified versions, please check the package file '$file' or specifiy --allow-no-version";
    } 
}



if($script ne ""){
    my $filename = "/tmp/install_packages_via_remotes.r";
    open(FH, '>', $filename) or die $!;
    print FH $script;
    close(FH);
    my $cmd = "$rscript_binary '$filename'";
    package_mgmt_utils::execute("$rscript_binary -e 'install.packages(\"remotes\",repos=\"https://cloud.r-project.org\")'",$dry_run);
    print "Executing:\n$script\n";
    package_mgmt_utils::execute($cmd,$dry_run);
    unlink($filename) or die "Can't delete $filename: $!\n";
}
