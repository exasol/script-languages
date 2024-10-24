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

The parser must correctly identify and handle Script Options with the syntax `%<optionKey><white spaces><optionValue>;`. It must also manage white spaces and ignore non-Script Options as specified.

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`

### Multiple and Options
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

The parser must remove found Script Options from the original script code.

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`
- `feat~java-specific-script-options~1`

### Java %scriptclass Option Handling
`req~java-scriptclass-option-handling~1`

The parser must correctly identify the first `%scriptclass` option and remove any additional occurrences of this option within the script code.

Needs: dsn

Covers:
- `feat~java-specific-script-options~1`

### Java %jar Option Handling
`req~java-jar-option-handling~1`

The parser must find multiple %jar options, handle duplicates properly, and ensure the values match the Java CLASSPATH environment variable syntax.

Needs: dsn

Covers:
- `feat~java-specific-script-options~1`

### Java %jvmoption Handling
`req~java-jvmoption-handling~1`

The parser must find multiple %jvmoption options, allowing duplicates and maintaining order.

Needs: dsn

Covers:
- `feat~java-specific-script-options~1`

### Java %import Option Handling
`req~java-import-option-replace-referenced-scripts~1`

For each found %import option, the parser must request and replace the referenced scripts recursively. This means,
if the referenced scripts contain also `%import` options, the implementation must replace those, too.

Needs: dsn

Covers:
- `feat~java-specific-script-options~1`

### Java %import Option Handling
`req~java-import-option-handling~1`

For each found %import option, the parser must handle nested Script Options appropriately:
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

Covers:
- `feat~general-script-options-parsing~1`

### Existing Parser Linker Namespace Compatibility
`req~existing-parser-linker-namespace-compatibility~1`

Ideally, the new parser should allow encapsulation in a custom C++ namespace, in order to avoid possible linker namespace conflicts with customer libraries.

Needs: dsn

Covers:
- `feat~general-script-options-parsing~1`

