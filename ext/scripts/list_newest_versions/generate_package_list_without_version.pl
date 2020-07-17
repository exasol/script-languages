#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS

  install_via_apt.pl [OPTIONS]
  Options:
    --help                Brief help message
    --file                Input file with each line represents a input. 
                          A line can have multiple elements separated by |. 
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
            "file=s" => \$file,
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
}


my $combining_template = '<<<<0>>>>';
my @separators = ("\n");
my @templates = ("<<<<0>>>>");

my $package_list = 
    package_mgmt_utils::generate_joined_and_transformed_string_from_file(
        $file,$element_separator,$combining_template,\@templates,\@separators);
print("$package_list\n");
