#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS
  Add a new PPA and clean up afterwards.

  install_ppa.pl [OPTIONS]
  Options:
    --help                Brief help message
    --dry-run             Doesn't execute the command, only prints it to STDOUT
    --ppa                 PPA to add.
    --out-file            The database file where the ppa info will be stored under /etc/apt/sources.list.d (without .list extension). This file must not exist.
=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use package_mgmt_utils;
use Getopt::Long;

my $help = 0;
my $dry_run = 0;
my $ppa = '';
my $out_file = '';


GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "ppa=s" => \$ppa,
            "out-file=s" => \$out_file
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;

if($ppa eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --ppa was not specified",1);
}

sub check_file {
    my $file_name = @_;
    die "out file for PPA: '$file_name' already exists!" if -e $file_name;
}

sub generate_install_command{
    my $ppa_name = '';
    if ($out_file ne '') {
        $ppa_name = $out_file;
    } else {
        #If out-file was not explicitly given, we try to assemble it based on the last part of the ppa.
        my @column_array = split / /, $ppa;
        $ppa_name = $column_array[-1];
        $ppa_name =~ s/^\s+|\s+$//g; #trim whitespaces
        $ppa_name =~ s/^\/|\/$//g;   #trim slash
        $ppa_name =~ s/^\\|\\$//g;   #trim backslash
    }
    $out_file = "/etc/apt/sources.list.d/$ppa_name.list";
    check_file($out_file);
    return "echo '$ppa' > $out_file";
}

my $cmd = generate_install_command();

package_mgmt_utils::execute("apt-get -y update",$dry_run);
package_mgmt_utils::execute("apt-get -y install ca-certificates",$dry_run); # Need ca-certificates for apt-get update after adding new ppa's
package_mgmt_utils::execute("apt-get -y clean", $dry_run);
package_mgmt_utils::execute("apt-get -y autoremove", $dry_run);
package_mgmt_utils::execute($cmd,$dry_run);
package_mgmt_utils::execute("apt-get -y update",$dry_run);
package_mgmt_utils::execute("apt-get -y clean", $dry_run);
package_mgmt_utils::execute("apt-get -y autoremove", $dry_run);
