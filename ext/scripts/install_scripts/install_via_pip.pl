#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS
  Installs python package via pip to the corresponding package site of the python binary

  install_via_pip.pl [OPTIONS]
  Options:
    --help               Brief help message
    --dry-run            Doesn't execute the command, only prints it to STDOUT
    --file               Input file with each line represents a input. 
                         A line can have multiple elements separated by --element-separator. 
                         Lines everything after a # is interpreted as comment
    --with-versions      Uses versions specified in the input file in the second element of each line
    --allow-no-versions  If --with-versions is active, allow packages to have no version specified
    --python-binary      Python-binary to use for the installation
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use utils;
use Getopt::Long;

my $help = 0;
my $dry_run = 0;
my $file = '';
my $python_binary = '';
my $with_versions = 0;
my $allow_no_version = 0;

GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "file=s" => \$file,
            "with-versions" => \$with_versions,
            "allow-no-version" => \$allow_no_version,
            "python-binary=s" => \$python_binary
          ) or utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
}

if($python_binary eq ''){
    utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --python-binary was not specified",1);
}

my $element_separator = '\\|';
my $combining_template = "$python_binary -m pip install --ignore-installed --progress-bar ascii --no-cache-dir <<<<0>>>>";
my @templates = ("'<<<<0>>>>'");
if($with_versions){
    @templates=("'<<<<0>>>>==<<<<1>>>>'")
}
my @separators = (" ");

my $cmd = 
    utils::generate_joined_and_transformed_string_from_file(
        $file,$element_separator,$combining_template,\@templates,\@separators);

if($with_versions and $allow_no_version){
    $cmd =~ s/==<<<<1>>>>//g;
}
if($with_versions and not $allow_no_version){
    if (index($cmd, "==<<<<1>>>>") != -1) {
        die "Command '$cmd' contains packages with unspecified versions, please check the package file '$file' or specifiy --allow-no-version";
    } 
}

if($cmd ne ""){
   utils::execute($cmd,$dry_run);
}
