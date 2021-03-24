#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS

  install_batch.pl [OPTIONS]
  Options:
    --help Brief help message
    --dry-run Doesn't execute the command, only prints it to STDOUT
    --file Input file with each line represents a input. 
           A line can have multiple elements separated by --element-separator. 
           Lines everything after a # is interpreted as comment
    --element-separator Separates elements in a line of the input file
    --combining-template Templates which combines all joined templates 
                         for the lines of the input file into one string.
                         The templates are addressed with <<<<0>>>>, ,,,
    --templates Pairs of template and separator pair. 
                The template is applied for each line in the input file and 
                than joined together with the separator.
                The elements of a line are addressed with <<<<0>>>>, ,,,
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use package_mgmt_utils;
use Getopt::Long;
use Pod::Usage;

my $help = 0;
my $dry_run = 0;
my $file = '';
my $element_separator = '';
my $combining_template = '';
my @template_separator_array = ();

GetOptions (
           "help" => \$help,
           "dry-run" => \$dry_run,
           "file=s" => \$file,
           "element-separator=s" => \$element_separator,
           "combining-template=s" => \$combining_template,
           "templates=s{2,}" => \@template_separator_array
         ) or pod2usage(2);
pod2usage(1) if $help;


if($file eq ''){
   pod2usage("Error in command line arguments: --file was not specified");
}
if($element_separator eq ''){
   pod2usage("Error in command line arguments: --element-separator was not specified");
}
if($combining_template eq ''){
   pod2usage("Error in command line arguments: --combining_template was not specified");
}


if($#template_separator_array < 1 and ($#template_separator_array+1) % 2 != 0){
   pod2usage("Error in command line arguments: --templates need to be specified in pairs of template and separator");
}
my @templates = ();
my @separators = ();
for (my $i = 0; $i < $#template_separator_array; $i += 2) {  
   push(@templates,$template_separator_array[$i]);
   push(@separators,$template_separator_array[$i+1]);
}

sub identity {
    my ($line) = @_;
    return $line 
}

my @rendered_line_transformation_functions = (\&identity);

my $cmd = 
   package_mgmt_utils::generate_joined_and_transformed_string_from_file(
       $file,$element_separator,$combining_template,\@templates,\@separators,\@rendered_line_transformation_functions);

package_mgmt_utils::execute("$cmd",$dry_run)
