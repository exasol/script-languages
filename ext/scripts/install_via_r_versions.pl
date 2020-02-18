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
    --rscript-binary Rscript binary to use for installation
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use utils;
use Getopt::Long;
use Pod::Usage;
#use IPC::System::Simple;

my $help = 0;
my $dry_run = 0;
my $file = '';
my $rscript_binary = '';

GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "file=s" => \$file,
            "rscript-binary=s" => \$rscript_binary
          ) or pod2usage(2);
pod2usage(1) if $help;


if($file eq ''){
    pod2usage("Error in command line arguments: --file was not specified");
}

if($rscript_binary eq ''){
    pod2usage("Error in command line arguments: --rscript-binary was not specified");
}

my $element_separator = '\\|';
my $combining_template = "$rscript_binary --default-packages 'versions' -e 'install.versions(c(<<<<0>>>>),c(<<<<1>>>>))'";
my @templates = ('"<<<<0>>>>"','"<<<<1>>>>"');
my @separators = (",",",");

my $cmd = 
    utils::generate_joined_and_transformed_string_from_file(
        $file,$element_separator,$combining_template,\@templates,\@separators);

utils::execute("$rscript_binary -e 'install.packages(\"install.packages('versions')\")'",$dry_run);
utils::execute($cmd,$dry_run);
