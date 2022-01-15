#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS
  Installs python package via pip to the corresponding package site of the python binary

  install_via_pip.pl [OPTIONS]
  Options:
    --help                                Brief help message
    --dry-run                             Doesn't execute the command, only prints it to STDOUT
    --file                                Input file with each line represents a input. 
                                          A line can have multiple elements separated by --element-separator. 
                                          Lines everything after a # is interpreted as comment
    --with-versions                       Uses versions specified in the input file in the second element of each line
    --allow-no-version                    If --with-versions is active, allow packages to have no version specified
    --allow-no-version-for-urls           If --with-versions is active, allow packages specified by urls to have no version
    --ignore-installed                    Set the --ignore-installed option for pip
    --use-deprecated-legacy-resolver      Set the --use-deprecated=legacy-resolver option for pip
    --python-binary                       Python-binary to use for the installation
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use package_mgmt_utils;
use Getopt::Long;

my $help = 0;
my $dry_run = 0;
my $file = '';
my $python_binary = '';
my $with_versions = 0;
my $allow_no_version = 0;
my $allow_no_version_for_urls = 0;
my $ignore_installed = 0;
my $use_deprecated_legacy_resolver = 0;
GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "file=s" => \$file,
            "with-versions" => \$with_versions,
            "allow-no-version" => \$allow_no_version,
            "allow-no-version-for-urls" => \$allow_no_version_for_urls,
            "ignore-installed" => \$ignore_installed,
            "use-deprecated-legacy-resolver" => \$use_deprecated_legacy_resolver,
            "python-binary=s" => \$python_binary
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
}

if($python_binary eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --python-binary was not specified",1);
}

my @pip_parameters = ();

if($ignore_installed){
  push @pip_parameters, "--ignore-installed";
}

if($use_deprecated_legacy_resolver){
  push @pip_parameters, "--use-deprecated=legacy-resolver";
}

my $pip_parameters_str = join( ' ', @pip_parameters);
my $element_separator = '\\|';
my $combining_template = "$python_binary -m pip install $pip_parameters_str --no-cache-dir <<<<0>>>>";
my @templates = ("'<<<<0>>>>'");
if($with_versions){
    @templates=("'<<<<0>>>>==<<<<1>>>>'")
}
my @separators = (" ");

sub identity {
    my ($line) = @_;
    return $line 
}


sub replace_missing_version{
    my ($line) = @_;
    $line =~ s/==<<<<1>>>>//g;
    return $line;
}

sub replace_missing_version_for_urls{
    my ($line) = @_;
    $line =~ s/([a-z+]+:\/\/.*)==<<<<1>>>>/$1/g;
    return $line;
}

my @rendered_line_transformation_functions = (\&identity);
if($with_versions and $allow_no_version){
    @rendered_line_transformation_functions = (\&replace_missing_version);
}elsif($with_versions and $allow_no_version_for_urls){
    @rendered_line_transformation_functions = (\&replace_missing_version_for_urls);
}

my $cmd = 
    package_mgmt_utils::generate_joined_and_transformed_string_from_file(
        $file, $element_separator, $combining_template, \@templates, \@separators, \@rendered_line_transformation_functions);

if($with_versions){
    if (index($cmd, "==<<<<1>>>>") != -1) {
        die "Command '$cmd' contains packages with unspecified versions, please check the package file '$file' or specifiy --allow-no-version";
    } 
}

if($cmd ne ""){
   package_mgmt_utils::execute($cmd,$dry_run);
}
