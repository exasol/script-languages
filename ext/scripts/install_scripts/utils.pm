use strict;

package utils;

#use Data::Dumper;

sub generate_joined_and_transformed_string_from_file{
    my ($file, $element_separator, $combining_template, $templates_ref, $separators_ref) = @_;
    my @templates = @$templates_ref;
    my @separators = @$separators_ref;
    my @elements_of_lines = get_lines_with_elements_from_file($file, $element_separator);
    my @transformed_lines = map(fill_template_for_lines($_,\@elements_of_lines),@templates);
    my $count_transformed_lines=scalar @transformed_lines;
    if($count_transformed_lines == 0){
        return "";
    }
    my $final_string = fill_template_with_joined_lines_of_elements($combining_template, \@separators, \@transformed_lines);
    return $final_string;
}

sub get_lines_with_elements_from_file{
    my ($file, $element_separator) = @_;
    my @lines = get_lines_from_commented_file($file);
    my @elements_of_lines = map(parse_lines_to_elements($element_separator,$_),@lines);
    return @elements_of_lines;
}

sub get_lines_from_commented_file{
    my ($file) = @_;
    my @lines = read_package_file($file);
    @lines = grep { $_ ne '' } @lines;
    @lines = map(remove_comments($_),@lines); 
    @lines = grep { $_ ne '' } @lines;
    return @lines;
}

sub read_package_file{
    my ($filename) = @_;
    # supporting only ascii files, because :encoding(UTF-8) is not per default available in ubuntu docker image
    open(my $fh, '<', $filename) 
      or die "Could not open file '$filename' $!"; 
    chomp(my @lines = <$fh>);
    close($fh)
      or die "Could not close file '$filename' $!"; 
    return @lines
}

sub remove_comments{
    my ($line) = @_;
    # check for: <whitespaces><content><whitespace><#comment>
    my $comment_start = "#";
    if($line =~ /^[ \t]*$comment_start.*$/g){
        return ""
    } elsif ($line =~ /^[ \t]*([^ \ŧ]+)([ \t]$comment_start.*)?[ \t]*$/g) {
        my $line_without_comments = $1;
        return $line_without_comments;
    } else {
        die "'$line' doesn't match regex"
    }
}

sub parse_lines_to_elements{
    my ($element_separator, $line) = @_;
    my @elements = split /$element_separator/, $line;
    return \@elements
}

sub fill_template_for_lines{
    my ($template, $elements_for_lines_ref) = @_;
    my @elements_for_lines = @$elements_for_lines_ref;
    my @filled_template_for_lines = map(fill_template($template,$_),@elements_for_lines);
    if(scalar @filled_template_for_lines == 0){
        return ();
    }else{
        return \@filled_template_for_lines;
    }
}

sub fill_template{
    my ($template,$elements_ref) = @_;
    my @elements=@$elements_ref;
    my $filled_template=$template;
    for (my $i=0; $i <= $#elements; $i++) {
        my $pattern = "<<<<$i>>>>";
        my $replace = "$elements[$i]";
        $filled_template=replace($pattern,$replace,$filled_template);
    }
    return $filled_template;
}

sub replace {
    my ($from,$to,$string) = @_;
    $string =~s/$from/$to/g;
    return $string;
}


sub fill_template_with_joined_lines_of_elements{
    my ($combining_template, $separators_ref, $lines_ref) = @_;
    my @separators = @$separators_ref;
    my @lines = @$lines_ref;

    my @joined_lines = map(join_lines($separators[$_], $lines[$_]), 0 .. $#lines);
    my $final_string = fill_template($combining_template, \@joined_lines);
    return $final_string;
}

sub join_lines{
    my ($separator, $lines_ref) = @_;
    my @lines = @$lines_ref;
    
    my $joined_lines = join($separator,@lines);
    return $joined_lines;
}

sub execute{
    my ($cmd, $dry_run) = @_;
    if($dry_run == 0){
        print("Executing: $cmd\n");
        my $status = system($cmd);
        if (($status >>=8) != 0) {
            die "Failed to run $cmd";
        }
    }else{
        print("Dry-Run: $cmd\n");
    }
}

sub extract_synopsis{
    my ($filename) = @_;
    # supporting only ascii files, because :encoding(UTF-8) is not per default available in ubuntu docker image
    open(my $fh, '<', $filename) 
      or die "Could not open file '$filename' $!"; 
    my @lines = <$fh>;
    close($fh)
      or die "Could not close file '$filename' $!"; 
    my $start = -1;
    my $end = -1;
    for(my $i = 0; $i <= $#lines; $i++){
      if($lines[$i] =~ "^=head1 SYNOPSIS"){
          if($start==-1){
              $start=$i;
              next;
          }else{
              die("Found SYNOPSIS twice!");
          }
      }
      if($lines[$i] =~ "^=cut"){
          if($start!=-1){
              $end=$i;
              last;
          }else{
              die("Found =cut before SYNOPSIS!");
          } 
      }
    }
    return join("",@lines[$start+1..$end-1]);
}

sub print_usage_and_abort{
    my ($filename,$message,$exitcode) = @_;
    my $usage=extract_synopsis($filename);
    print("$message\n");
    print("\n");
    print($usage);
    exit($exitcode);
}

1;

