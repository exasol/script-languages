#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS

  install_batch.pl [OPTIONS]
  Options:
    --help               Brief help message
    --file1              Input file with each line represents a input. 
                         A line can have multiple elements separated by --element-separator. 
                         Lines everything after a # is interpreted as comment
    --file2              Input file with each line represents a input. 
                         A line can have multiple elements separated by --element-separator. 
                         Lines everything after a # is interpreted as comment
    --element-separator  Separates elements in a line of the input file
                                     
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use utils;
use Getopt::Long;
use Data::Dumper;

my $help = 0;
my $file1 = '';
my $file2 = '';
my $element_separator = '\\|';

GetOptions (
           "help" => \$help,
           "file1=s" => \$file1,
           "file2=s" => \$file2,
           "element-separator=s" => \$element_separator,
         ) or utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
utils::print_usage_and_abort(__FILE__,"",0) if $help;

if($file1 eq ''){
    utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file1 was not specified",1);
}
if($file1 eq ''){
    utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --file2 was not specified",1);
}


my @elements_of_lines_file1 = utils::get_lines_with_elements_from_file($file1, $element_separator);
my @elements_of_lines_file2 = utils::get_lines_with_elements_from_file($file2, $element_separator);

sub generate_hashmap{
    my ($elements_of_lines_ref) = @_;
    my @elements_of_lines = @$elements_of_lines_ref;
    
    my %package_map;
    foreach (@elements_of_lines){
       my $elements_ref = $_;
       my @elements = @$elements_ref;
       my $package_name = $elements[0];
       my $package_version = $elements[1];
       $package_map{$package_name}=$package_version;
    }
    return %package_map;
}

my %package_map_file1 = generate_hashmap(\@elements_of_lines_file1);
my %package_map_file2 = generate_hashmap(\@elements_of_lines_file2);

my %joined_package_map;
foreach my $package_name (keys %package_map_file1) {
    if(exists($package_map_file2{$package_name})){
        my $version1=$package_map_file1{$package_name};
        my $version2=$package_map_file2{$package_name};
        if(defined($version1) != defined($version2) or 
            (defined($version1) and  defined($version2) and  
			    $version1 ne $version2)){
            my @versions=($version1,$version2);
            $joined_package_map{$package_name}=\@versions; 
        }
    }else{ 
        my @versions=($package_map_file1{$package_name},"package not found");
        $joined_package_map{$package_name}=\@versions;
    }
}

$element_separator =~ s/\\//ig;
foreach my $package_name (keys %joined_package_map){
    my $versions_ref=$joined_package_map{$package_name};
    my @versions=@$versions_ref;
    my $version1=$versions[0];
    if(not defined($version1)){
    	$version1="version not specified";
    }
    my $version2=$versions[1];
    if(not defined($version2)){
    	$version2="version not specified";
    }
    print("$package_name$element_separator$version1$element_separator$version2\n");
}

#print(Dumper(\%joined_package_map));
