#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS

  install_via_apt.pl [OPTIONS]
  Options:
    --help                Brief help message
    --dry-run             Doesn't execute the command, only prints it to STDOUT
    --with-versions       Uses versions specified in the input file in the second element of each line
    --element-separator   Element separator regex in a line in the input file, defaults to "|"
    --mark-hold           Execute apt-mark hold for the package in the input file after installation
    --file                Input file with each line represents a input. 
                          A line can have multiple elements separated by --element-separator. 
                          Lines everything after a # is interpreted as comment.
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use utils;
use Getopt::Long;

my $help = 0;
my $dry_run = 0;
my $file = '';
my $element_separator = "\\|";
my $with_versions = 0;
my $mark_hold = 0;

GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "file=s" => \$file,
            "element-separator=s" => \$element_separator,
            "with-versions" => \$with_versions,
            "mark-hold" => \$mark_hold
          ) or utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
utils::print_usage_and_abort(__FILE__,"",0) if $help;


if($file eq ''){
    utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file was not specified",1);
}


sub generate_install_command{
    my ($element_separator,$with_versions) = @_;
    my $combining_template = 'apt-get install -V -y --no-install-recommends <<<<0>>>>';
    my @separators = (" ");
    my @templates = ("'<<<<0>>>>'");
    print("$with_versions\n");
    if($with_versions){
        @templates=("'<<<<0>>>>=<<<<1>>>>'")
    }

    my $cmd = 
        utils::generate_joined_and_transformed_string_from_file(
            $file,$element_separator,$combining_template,\@templates,\@separators);
    return $cmd;
}


sub generate_mark_command{ 
    my ($element_separator) = @_;
    my $combining_template = 'apt-mark hold <<<<0>>>>';
    my @templates = ("'<<<<0>>>>'");
    my @separators = (" ");

    my $cmd = 
        utils::generate_joined_and_transformed_string_from_file(
            $file,$element_separator,$combining_template,\@templates,\@separators);
    return $cmd;
}

my $install_cmd = generate_install_command($element_separator,$with_versions);
my $mark_cmd = generate_mark_command($element_separator);

utils::execute("apt-get -y update",$dry_run);
utils::execute($install_cmd,$dry_run);
if($mark_hold){
    utils::execute($mark_cmd,$dry_run);
}
utils::execute("locale-gen en_US.UTF-8",$dry_run);
utils::execute("update-locale LC_ALL=en_US.UTF-8",$dry_run);
utils::execute("apt-get -y clean",$dry_run);
utils::execute("apt-get -y autoremove",$dry_run);
utils::execute("ldconfig",$dry_run);
