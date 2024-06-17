use strict;
use warnings;

package package_mgmt_utils;

#use Data::Dumper;

sub generate_joined_and_transformed_string_from_file{
    my ($file, $element_separator, $combining_template, $templates_ref, $separators_ref, $rendered_line_transformation_functions_ref) = @_;
    my @transformed_lines = generate_transformed_lines_for_templates(
        $file, $element_separator, $templates_ref, $rendered_line_transformation_functions_ref);
    my $final_string = generate_joined_string_from_lines(\@transformed_lines, $combining_template,, $separators_ref);
    return $final_string;
}


sub generate_joined_and_transformed_string_from_files {
    my ($element_separator, $combining_template, $templates_ref, $separators_ref, $rendered_line_transformation_functions_ref, $files_ref) = @_;
    my @transformed_lines;
    foreach my $file ( @$files_ref ) {
        my @transformed_lines_for_current_file = generate_transformed_lines_for_templates(
            $file, $element_separator, $templates_ref, $rendered_line_transformation_functions_ref);

        if (!@transformed_lines) {
            @transformed_lines = @transformed_lines_for_current_file;
        } else {
            if ($#transformed_lines_for_current_file != $#transformed_lines) {
                die "Internal error processing package file $file\n";
            }
            for my $i (0 .. $#transformed_lines_for_current_file) {
                #Resolve reference for the resulting and new arrays, merge both and assign back the reference to the resulting array
                my $transformed_lines_for_current_file_part_ref = $transformed_lines_for_current_file[$i];
                my @transformed_lines_for_current_file_part = @$transformed_lines_for_current_file_part_ref;

                my $transformed_lines_part_ref = $transformed_lines[$i];
                my @transformed_lines_part = @$transformed_lines_part_ref;

                push (@transformed_lines_part, @transformed_lines_for_current_file_part);
                $transformed_lines[$i] = \@transformed_lines_part;
            }
        }
    }
    my $final_string = generate_joined_string_from_lines(\@transformed_lines, $combining_template, $separators_ref);
    return $final_string;
}

sub generate_joined_string_from_lines{
    my ($lines_ref, $combining_template, $separators_ref) = @_;
    my @separators = @$separators_ref;
    my @lines = @$lines_ref;
    my $count_lines=scalar @lines;
    if($count_lines == 0){
        return "";
    }
    my $final_string = fill_template_with_joined_lines_of_elements($combining_template, \@separators, \@lines);
    return $final_string;
}

sub generate_transformed_lines_for_templates{ 
    my ($file, $element_separator, $templates_ref, $rendered_line_transformation_functions_ref) = @_;
    my @templates = @$templates_ref;
    my @rendered_line_transformation_functions = @$rendered_line_transformation_functions_ref;
    my @elements_of_lines = get_lines_with_elements_from_file($file, $element_separator);
    my @template_transformation_function_pairs = zip_two_arrays($templates_ref,$rendered_line_transformation_functions_ref);
    my @transformed_lines = map(generate_transformed_lines_for_template(\@elements_of_lines,$_),@template_transformation_function_pairs);
    return @transformed_lines;
}

sub min{
    my ($x, $y) = @_;
    return $x <= $y ? $x : $y;
}

sub zip_two_arrays{
    my ($array1_ref, $array2_ref) = @_;
    my @array1 =  @$array1_ref;
    my @array2 =  @$array2_ref;
    my @result = ();
    my $min_size = min($#array1, $#array2);
    for my $i (0 .. $min_size) {
        my @element = ($array1[$i], $array2[$i]);
        push (@result, \@element); 
    }
    return @result;
}

sub generate_transformed_lines_for_template{
    my ($elements_of_lines_ref, $template_transformation_function_pair_ref) = @_;
    my @elements_of_lines=@$elements_of_lines_ref;
    my @template_transformation_function_pair = @$template_transformation_function_pair_ref;
    my $template=$template_transformation_function_pair[0];
    my $transformation_function_ref=$template_transformation_function_pair[1];
    my @rendered_lines = fill_template_for_lines($template,\@elements_of_lines);
    my @transformed_lines = map(apply_rendered_line_transformation_function($_, $transformation_function_ref),@rendered_lines);
    return @transformed_lines; 
}

sub apply_rendered_line_transformation_function{
    my ($lines_ref, $transformation_function_ref) = @_;
    my @lines = @$lines_ref;
    my @transformed_lines = map($transformation_function_ref->($_),@lines);
    return \@transformed_lines;
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

sub find_files_matching_pattern {
    my ($dir, $pattern) = @_;
    my @files;

    opendir(my $dh, $dir) or die "Can't open directory $dir: $!";
    while (my $file = readdir($dh)) {
        next if ($file =~ /^\./);  # Skip hidden files
        my $path = "$dir/$file";
        if (-d $path) {
            # Recursively traverse subdirectories
            push(@files, find_files_matching_pattern($path, $pattern));
        } elsif (-f $path and $file =~ /^$pattern$/) {
            push(@files, $path);
        }
    }
    closedir($dh);

    return @files;
}

sub merge_package_files {
    my ($base_file, $base_search_dir, $file_pattern) = @_;

    my @files_in_search_dir = find_files_matching_pattern($base_search_dir, $file_pattern);
    return sort ($base_file, @files_in_search_dir);
}

1;