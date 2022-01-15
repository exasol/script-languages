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


my $combining_template = "library(remotes)\n<<<<0>>>>";
my @separators = ("\n");
my @templates = ('install_version("<<<<0>>>>",NULL,repos="http://cran.r-project.org", Ncpus=4)');
if($with_versions){  
    @templates = ('install_version("<<<<0>>>>","<<<<1>>>>",repos="http://cran.r-project.org", Ncpus=4)');
}

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

my $script = 
    package_mgmt_utils::generate_joined_and_transformed_string_from_file(
        $file,$element_separator,$combining_template,\@templates,\@separators,\@rendered_line_transformation_functions);

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
    package_mgmt_utils::execute("$rscript_binary -e 'install.packages(\"remotes\",repos=\"http://cran.r-project.org\")'",$dry_run);
    print "Executing:\n$script\n";
    package_mgmt_utils::execute($cmd,$dry_run);
    unlink($filename) or die "Can't delete $filename: $!\n";
}
