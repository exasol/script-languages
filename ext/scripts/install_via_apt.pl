#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS

  install_via_apt.pl [OPTIONS]
  Options:
    --help Brief help message
    --dry-run Doesn't execute the command, only prints it to STDOUT
    --file Input file with each line represents a input. 
           A line can have multiple elements separated by --element-separator. 
           Lines everything after a # is interpreted as comment
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use utils;
use Getopt::Long;

my $help = 0;
my $dry_run = 0;
my $file = '';

GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "file=s" => \$file,
          ) or utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
}

my $element_separator = '\\|\\|\\|\\|\\|\\|\\|\\|';
my $combining_template = 'apt-get install -y --no-install-recommends <<<<0>>>>';
my @templates = ("'<<<<0>>>>'");
my @separators = (" ");

my $cmd = 
    utils::generate_joined_and_transformed_string_from_file(
        $file,$element_separator,$combining_template,\@templates,\@separators);

utils::execute("apt-get -y update",$dry_run);
utils::execute($cmd,$dry_run);
utils::execute("locale-gen en_US.UTF-8",$dry_run);
utils::execute("update-locale LC_ALL=en_US.UTF-8",$dry_run);
utils::execute("apt-get -y clean",$dry_run);
utils::execute("apt-get -y autoremove",$dry_run);
utils::execute("ldconfig",$dry_run);
