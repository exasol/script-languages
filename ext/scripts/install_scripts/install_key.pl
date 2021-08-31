#!/usr/bin/perl -w

=pod
 
=head1 SYNOPSIS
  Add a new key to apt.

  install_ppa.pl [OPTIONS]
  Options:
    --help                Brief help message
    --dry-run             Doesn't execute the command, only prints it to STDOUT
    --key                 Key to add.
    --key-server          Key-server belonging to key (for example: hkp://keyserver.ubuntu.com:80).

=cut

use strict;
use File::Basename;
use lib dirname (__FILE__);
use package_mgmt_utils;
use Getopt::Long;

my $help = 0;
my $dry_run = 0;
my $key_server = '';
my $key = '';


GetOptions (
            "help" => \$help,
            "dry-run" => \$dry_run,
            "key=s" => \$key,
            "key-server=s" => \$key_server
          ) or package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments",2);
package_mgmt_utils::print_usage_and_abort(__FILE__,"",0) if $help;

if($key eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --key was not specified",1);
}

if($key_server eq ''){
    package_mgmt_utils::print_usage_and_abort(__FILE__,"Error in command line arguments: --key-server was not specified",1);
}

package_mgmt_utils::execute("apt-get -y update",$dry_run);
package_mgmt_utils::execute("apt-get install -y dirmngr",$dry_run);
package_mgmt_utils::execute("gpg --keyserver $key_server --recv-keys $key",$dry_run);
package_mgmt_utils::execute("gpg -a --export $key | apt-key add -",$dry_run);
