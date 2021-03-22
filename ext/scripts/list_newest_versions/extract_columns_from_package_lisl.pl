#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS

  install_via_apt.pl [OPTIONS]
  Options:
    --help                      Brief help message
    --file                      Input file with each line represents a input. 
                                A line can have multiple elements separated by |. 
                                Lines everything after a # is interpreted as comment.
    --columns                   Indices of columns to extract starting with 0 seperated by ",", e.g --column "0,3,1"
    --output-column-seperator   Seperator between columns used for the output. Default: |
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use package_mgmt_utils;
use Getopt::Long;
use List::Util qw(all);

my $help = 0;
my $dry_run = 0;
my $file = '';
my $element_separator = "\\|";
my $with_versions = 0;
my $mark_hold = 0;
my $allow_no_version = 0;
my $columns = '';
my $output_column_separator = "\\|";

GetOptions (
            "help" => \$help,
            "file=s" => \$file,
            "columns=s" => \$columns,
            "output-column-separator=s" => \$output_column_separator
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
}

if($columns eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --columns was not specified or empty",1);
}

my $combining_template = '<<<<0>>>>';
my @separators = ("\n");


sub generate_template{
    my ($separator, $columns) = @_;
    my @column_array = split /,/, $columns;
    unless(all { $_ =~ /^\d+$/ } @column_array){
        package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --columns contains invalid entry",1);
    }
    my $template = join($separator, map { "<<<<$_>>>>" } @column_array);
    return $template;
}

my @templates = (generate_template($output_column_separator,$columns));

my $package_list = 
    package_mgmt_utils::generate_joined_and_transformed_string_from_file(
        $file,$element_separator,$combining_template,\@templates,\@separators);
print("$package_list\n");
