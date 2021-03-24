#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS

  install_via_apt.pl [OPTIONS]
  Options:
    --help                Brief help message
    --dry-run             Doesn't execute the command, only prints it to STDOUT
    --with-versions       Uses versions specified in the input file in the second element of each line
    --allow-no-versions   If --with-versions is active, allow packages to have no version specified
    --mark-hold           Execute apt-mark hold for the package in the input file after installation
    --file                Input file with each line represents a input. 
                          A line can have multiple elements separated by --element-separator. 
                          Lines everything after a # is interpreted as comment.
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use package_mgmt_utils;
use Getopt::Long;

my $help = 0;
my $dry_run = 0;
my $file = '';
my $element_separator = "\\|";
my $with_versions = 0;
my $mark_hold = 0;
my $allow_no_version = 0;

GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "file=s" => \$file,
            "with-versions" => \$with_versions,
            "allow-no-version" => \$allow_no_version,
            "mark-hold" => \$mark_hold
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
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

sub generate_install_command{
    my ($element_separator,$with_versions) = @_;
    my $combining_template = 'apt-get install -V -y --no-install-recommends <<<<0>>>>';
    my @separators = (" ");
    my @templates = ("'<<<<0>>>>'");
    if($with_versions){
        @templates=("'<<<<0>>>>=<<<<1>>>>'")
    }

    my @rendered_line_transformation_functions = (\&identity);
    if($with_versions and $allow_no_version){
        @rendered_line_transformation_functions = (\&replace_missing_version);
    }

    my $cmd = 
        package_mgmt_utils::generate_joined_and_transformed_string_from_file(
            $file,$element_separator,$combining_template,\@templates,\@separators,\@rendered_line_transformation_functions);
    
    if($with_versions and not $allow_no_version){
        if (index($cmd, "=<<<<1>>>>") != -1) {
            die "Command '$cmd' contains packages with unspecified versions, please check the package file '$file' or specifiy --allow-no-version";
        } 
    }
    return $cmd;
}


sub generate_mark_command{ 
    my ($element_separator) = @_;
    my $combining_template = 'apt-mark hold <<<<0>>>>';
    my @templates = ("'<<<<0>>>>'");
    my @separators = (" ");

    my @rendered_line_transformation_functions = (\&identity);
    my $cmd = 
        package_mgmt_utils::generate_joined_and_transformed_string_from_file(
            $file,$element_separator,$combining_template,\@templates,\@separators,\@rendered_line_transformation_functions);
    return $cmd;
}

my $install_cmd = generate_install_command($element_separator,$with_versions);
my $mark_cmd = generate_mark_command($element_separator);

if($install_cmd ne ""){
    package_mgmt_utils::execute("apt-get -y update",$dry_run);
    eval { package_mgmt_utils::execute($install_cmd,$dry_run) };
	my $script_dir = dirname (__FILE__);
	if($@){	
	    print("$@\n");
	}
	print("\n");
	print("Checking for new version of packages in '$file'\n");
    	package_mgmt_utils::execute("$script_dir/../list_newest_versions/list_newest_versions_for_apt.sh $file", $dry_run);
	if($@){
	    exit(1);
	}
    
    if($mark_hold && ($mark_cmd ne "")){
        package_mgmt_utils::execute($mark_cmd,$dry_run);
    }
    package_mgmt_utils::execute("locale-gen en_US.UTF-8",$dry_run);
    package_mgmt_utils::execute("update-locale LC_ALL=en_US.UTF-8",$dry_run);
    package_mgmt_utils::execute("apt-get -y clean",$dry_run);
    package_mgmt_utils::execute("apt-get -y autoremove",$dry_run);
    package_mgmt_utils::execute("ldconfig",$dry_run);
}
