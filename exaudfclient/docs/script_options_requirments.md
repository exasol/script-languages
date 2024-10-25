# System Requirement Specification

This document outlines the user-centric requirements for the new Exasol UDF Client Script Options parser. It identifies the key features, high-level requirements, and user roles to ensure clarity and alignment with user needs.

## Roles

This section lists the roles that will interact with or benefit from the parser system.

### Database Administrator
Database Administrators manage the database environment and ensure the efficient execution of UDFs within Exasol, configuring and overseeing script execution.

### Data Scientist
Data Scientists develop and deploy UDFs in languages such as Java, Python, or R to process and analyze data within Exasol.

## Features

This section lists the key features of the new UDF Client Script Options parser which you would highlight in a product leaflet.

### General Script Options Parsing
`feat~general-script-options-parsing~1`

Script Options must be parsed according to syntax definition.
Developers can add additional Options in an easy and consistent way.

Needs: req

### Java-specific Script Options
`feat~java-specific-script-options~1`

The parser must process all Java specific options correctly.

Needs: req

## High-level Requirements

This section details the high-level requirements for the new parser system, linked to the features listed above.

### General Script Options Parsing
`req~general-script-options-parsing~1`

The parser must correctly identify and handle Script Options with the syntax `%<optionKey><white spaces><optionValue>;`.
The separator between

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`

### White Spaces
`req~white-spaces~1`

The following is the list of white spaces:
|======================================================= 
| Name         | C/Python/Java | ASCII Dec | ASCII Hex | 
| tabulator    | '\t'          | 9         | 0x09      |
| vertical tab | '\v'          | 11         | 0x0b     |
| form feed    | '\f'          | 12         | 0x0c     |
| space        | ' '           | 30        | 0x20      |     
|======================================================= 

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`


### Leading White Spaces Options Parsing
`req~leading-white-spaces-script-options-parsing~1`

The parser must recognize Script Options for lines starting with white space characters. 

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`

Depends:
 - `req~white-spaces~1`

### Ignore anything which is not a Script Option
`req~ignore-none-script-options~1`

If there is any character in front of a Script Option which is not a white space, the parser must ignore the option(s). 

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`

Depends:
 - `req~white-spaces~1`

### Multiple Line Script Options Parsing
`req~multiple-lines-script-options-parsing~1`

The parser must recognize Script Options at any line in the given script code.

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`

### White Spaces Options Parsing V1
`req~white-spaces-script-options-parsing-v1~1`

All white spaces between the option key and option value are to be interpreted as separator. 
White spaces between the option value and the terminating ";" are to be removed from the option value.

Needs: dsn

Tags: V1

Covers:
- `feat~general-script-options-parsing~1`

Depends:
 - `req~white-spaces~1`

### White Spaces Options Parsing V2
`req~white-spaces-script-options-parsing-v2~1`

All white spaces between the option key and option value are to be ignored. The following rules for escape sequences at **the start** of a script optionValue are to be applied:
- '\ ' => space character
- '\t' => <tab> character
- '\f' => <form feed> character
- '\v' => <vertical tab> character

White spaces between the option value and the terminating ";" shall remain.

Needs: dsn

Tags: V2

Covers:
- `feat~general-script-options-parsing~1`

Depends:
 - `req~white-spaces~1`

### Multiple Options
`req~multiple-options-management~1`

The parser must collect multiple Script Options with the same key. Note: the specific handling depends on the option handler.

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`

### Duplicate Options Management
`req~multiple-options-management~1`

The parser must collect multiple Script Options with the same key and value. Note: the specific handling depends on the option handler.

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`

### Script Option Removal
`req~script-option-removal~1`

The parser handler must remove found known Script Options from the original script code. Successfully parsed Script Options, which are not recognized by the parser handler shall remain in the script code.

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`
- `feat~java-specific-script-options~1`


### Escape Sequence Script Options Parsing
`req~escape-sequence-script-options-parsing~1`

The following rules for escape sequences at any place within a script optionValue are to be applied:
- '\n' => <line feed> character
- '\r' => <carriage return> character
- '\;' => ';' character

Needs: dsn

Tags: V2

Covers:
- `feat~general-script-options-parsing~1`


### Java %scriptclass Option Handling V1
`req~java-scriptclass-option-handling-v1~1`

The Java parser handler must correctly identify the first `%scriptclass` option and remove only this single instance from the script code. Any further occurrences of `%scriptclass` option shall stay in the source script code.
The value should be handled according to the [Java specification for identifies](https://docs.oracle.com/javase/specs/jls/se7/html/jls-3.html#jls-3.8).

Needs: dsn

Tags: V1

Covers:
- `feat~java-specific-script-options~1`


### Java %scriptclass Option Handling V2
`req~java-scriptclass-option-handling-v2~1`

The Java parser handler must correctly identify the first `%scriptclass` option and remove any additional occurrences of this option within the script code.
The value should be handled according to the [Java specification for identifies](https://docs.oracle.com/javase/specs/jls/se7/html/jls-3.html#jls-3.8).

Needs: dsn

Tags: V2

Covers:
- `feat~java-specific-script-options~1`

### Java %jar Option Handling V1
`req~java-jar-option-handling-v1~1`

The Java parser handler must find multiple %jar options. The values are to be interpreted as the Java CLASSPATH: `<file1>:<file2>:...:<filen>`.
The Java parser handler shall split the entries by the colon character. The Java parser handler shall identify duplicated files and order the result of all `%jar` options alphabetically.

Needs: dsn

Tags: V1

Covers:
- `feat~java-specific-script-options~1`

### Java %jar Option Handling V2
`req~java-jar-option-handling-v2~1`

The Java parser handler must find multiple %jar options. The values are to be interpreted as the Java CLASSPATH: `<file1>:<file2>:...:<filen>`.
The Java parser handler shall split the entries by the colon character. The Java parser handler must keep duplicates. The order of the entries must not change.

Needs: dsn

Tags: V2

Covers:
- `feat~java-specific-script-options~1`

### Java %jar Option Trailing White Space Handling
`req~java-jar-option-trailing-white-space-handling~1`

The Java parser handler must remove trailing white spaces for `%jar` option values if they are part of the escape sequence '\ '. Escape sequences at the end of a found `%jar` option of the form `\ ` must be replaced with ' '.
This approach provides backwards compatibility for most existing UDF's from customers.

Needs: dsn

Tags: V2

Covers:
- `feat~java-specific-script-options~1`

Depends: 
- `req~white-spaces-script-options-parsing-v2~1`

### Java %jvmoption Handling
`req~java-jvmoption-handling~1`

The Java parser handler must find multiple %jvmoption options, allowing duplicates and maintaining order.

Needs: dsn

Covers:
- `feat~java-specific-script-options~1`

### Java %import Option Replace Referenced Sripts
`req~java-import-option-replace-referenced-scripts~1`

For each found %import option, the Java parser handler must request and replace the referenced scripts recursively. This means,
if the referenced scripts contain also `%import` options, the implementation must replace those, too.
The referenced script name should be handled according to the [Exasol SQL identifier specification](https://docs.exasol.com/db/latest/sql_references/basiclanguageelements.htm#SQLidentifier).

Needs: dsn

Covers:
- `feat~java-specific-script-options~1`

### Java %import Option Handling
`req~java-import-option-handling~1`

For each found %import option, the Java parser handler must handle nested Script Options appropriately:
1. `%scriptclass` option must be ignored, but removed from the script code.
2. All other options must be handled as if they were part of the source script.

Needs: dsn

Covers:
- `feat~java-specific-script-options~1`

### Existing Parser Library Dependencies
`req~existing-parser-library-dependencies~1`

The new parser must be implemented using an existing, open-source parser that supports definition of Lexer and Parser Rules in C++ code without additional runtime dependencies.
The implementation needs to be Open Source because the projects where the parser will be used are mainly Open Source, too. 
It is important to avoid additional runtime dependencies, as this would complicate the setup and maintenance of the runtime environment of the UDF client (aka Script Languages Container).


Needs: dsn

Tags: V2

Covers:
- `feat~general-script-options-parsing~1`

### Existing Parser Linker Namespace Compatibility
`req~existing-parser-linker-namespace-compatibility~1`

Ideally, the new parser should allow encapsulation in a custom C++ namespace, in order to avoid possible linker namespace conflicts with customer libraries.

Needs: dsn

Tags: V2

Covers:
- `feat~general-script-options-parsing~1`

