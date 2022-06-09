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
    --channel-file                                File which specifies the conda channels to use for installation. 
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
my $channel_file = '';
my $python_binary = '';
my $with_versions = 0;
my $allow_no_version = 0;
GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "file=s" => \$file,
	    "channel-file=s" => \$channel_file,
            "with-versions" => \$with_versions,
            "allow-no-version" => \$allow_no_version
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
}

if($channel_file eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --channel-file was not specified",1);
}

sub identity {
    my ($line) = @_;
    return $line 
}

sub replace_missing_version{
    my ($line) = @_;
    $line =~ s/==<<<<1>>>>//g;
    return $line;
}

sub replace_line_for_missing_version{
    my ($line) = @_;
    $line =~ s/^.*==<<<<1>>>>$//g;
    return $line;
}

sub generate_pinned_file{
    my ($file, $dry_run, $with_versions, $allow_no_version) = @_;
    my $element_separator = '\\|';
    my $combining_template = "<<<<0>>>>";
    my @templates=("'<<<<0>>>> ==<<<<1>>>>'");
    my @separators = ("\n");

    my @rendered_line_transformation_functions = (\&identity);
    if($with_versions and $allow_no_version){
        @rendered_line_transformation_functions = (\&replace_line_for_missing_version);
    }

    my $pinned_packages_file = 
        package_mgmt_utils::generate_joined_and_transformed_string_from_file(
            $file, $element_separator, $combining_template, \@templates, \@separators, \@rendered_line_transformation_functions);

    if($with_versions){
        if (index($pinned_packages_file, "==<<<<1>>>>") != -1) {
            die "Pinned package file contains packages with unspecified versions, please check the package file '$file' or specifiy --allow-no-version. \n $pinned_packages_file";
        } 
    }
    my $filename = '/opt/conda/conda-meta/pinned';
    if($dry_run ne 0){
        open(my $fh, '>>', $filename) or die "Could not open file '$filename' $!";
        print $fh "$pinned_packages_file";
        close $fh;
    }else{
        print "Generated following content for $filename\n";
        print "$pinned_packages_file\n";
    }
}

sub generate_channel_args{
    my ($channel_file) = @_;

    my $element_separator = '\\|';
    my $combining_template = "<<<<0>>>>";
    my @templates=("-c '<<<<0>>>>'");
    my @separators = (" ");
    my @rendered_line_transformation_functions = (\&identity);
    my $channel_args = 
        package_mgmt_utils::generate_joined_and_transformed_string_from_file(
            $channel_file, $element_separator, $combining_template, \@templates, \@separators, 
	    \@rendered_line_transformation_functions);
    return $channel_args
}

sub run_install_command{
    my ($file, $dry_run, $with_versions, $allow_no_version, $channel_args) = @_;
    my $element_separator = '\\|';
    my $combining_template = "/bin/micromamba --yes install --freeze-installed $channel_args <<<<0>>>>";
    my @templates=("'<<<<0>>>>==<<<<1>>>>'");
    my @separators = (" ");

    my @rendered_line_transformation_functions = (\&identity);
    if($with_versions and $allow_no_version){
        @rendered_line_transformation_functions = (\&replace_missing_version);
    }

    my $cmd = 
        package_mgmt_utils::generate_joined_and_transformed_string_from_file(
            $file, $element_separator, $combining_template, \@templates, \@separators, \@rendered_line_transformation_functions);

    if($with_versions){
        if (index($cmd, "==<<<<1>>>>") != -1) {
            die "Command '$cmd' contains packages with unspecified versions, please check the package file '$file' or specifiy --allow-no-version.";
        } 
    }
    if($cmd ne ""){
       package_mgmt_utils::execute($cmd,$dry_run);
       package_mgmt_utils::execute("/bin/micromamba clean --all --yes",$dry_run);
    }
}


generate_pinned_file($file, $dry_run, $with_versions, $allow_no_version);
my $channel_args = generate_channel_args($channel_file);
run_install_command($file, $dry_run, $with_versions, $allow_no_version, $channel_args);
