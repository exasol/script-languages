#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS
  Add a new PPA and clean up afterwards.

  install_ppa.pl [OPTIONS]
  Options:
    --help                Brief help message
    --dry-run             Doesn't execute the command, only prints it to STDOUT
    --ppa                 PPA to add.

=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use package_mgmt_utils;
use Getopt::Long;

my $help = 0;
my $dry_run = 0;
my $ppa = '';


GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "ppa=s" => \$ppa
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;

if($ppa eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --ppa was not specified",1);
}

print("PPA is $ppa");
my $cmd = "add-apt-repository -y '$ppa'";

my $ppa_provider = "software-properties-common";
package_mgmt_utils::execute("apt-get -y update",$dry_run);
package_mgmt_utils::execute("apt-get install -y $ppa_provider",$dry_run);
package_mgmt_utils::execute($cmd,$dry_run);
package_mgmt_utils::execute("apt-get -y --purge autoremove $ppa_provider",$dry_run);
package_mgmt_utils::execute("locale-gen en_US.UTF-8",$dry_run);
package_mgmt_utils::execute("update-locale LC_ALL=en_US.UTF-8",$dry_run);
package_mgmt_utils::execute("apt-get -y update",$dry_run);
package_mgmt_utils::execute("apt-get check",$dry_run);
package_mgmt_utils::execute("apt-get -y -f install",$dry_run);
package_mgmt_utils::execute("apt-get -y autoclean",$dry_run);
package_mgmt_utils::execute("ldconfig",$dry_run);

