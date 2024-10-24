# Design Document

This document details the design aspects of the new Exasol UDF Client Script Options parser, based on the high-level requirements outlined in the System Requirement Specification.

## Introduction

The purpose of this document is to provide an in-depth implementation view of the UDF Client Script Options parser, ensuring it meets all specified high-level requirements and integrates seamlessly into the existing Exasol environment. The parser needs to be implemented in C++.

## Constraints

- The parser implementation must be in C++.
- The chosen parser implementation is [ctpg](https://github.com/peter-winter/ctpg), which supports the definition of Lexer and Parser Rules in C++ code.

## Design Requirements

This section outlines the specific design requirements linked to the high-level requirements, detailing the implementation and methods to achieve them.

### Parser Implementation
`dsn~parser-implementation~1`

Implement the parser using [ctpg](https://github.com/peter-winter/ctpg), an open-source parser library. This library will be used to define Lexer and Parser Rules in C++ code, ensuring no additional runtime dependencies exist.

Needs: req
Covers:
- `req~new-parser-implementation~1`

### Lexer and Parser Rules
`dsn~lexer-parser-rules~1`

Define the Lexer rules to tokenize `%optionKey`, `optionValue`, and allowed whitespace characters, including `\t`, `\v`, and `\f`. The Parser rules will define the grammar to correctly identify Script Options, manage multiple options with the same key, and handle duplicates.

Needs: req
Covers:
- `req~general-script-options-parsing~1`

### Handling Multiple and Duplicate Options
`dsn~handling-multiple-duplicate-options~1`

Create a mechanism within the parser to collect and aggregate multiple Script Options with the same key. Use a data structure (e.g., a map of lists) to store options by key and ensure duplicates are handled according to specified rules for different options.

Needs: req
Covers:
- `req~multiple-duplicate-options-management~1`

### Script Option Removal Mechanism
`dsn~script-option-removal-mechanism~1`

Implement a method to remove identified Script Options from the original script code. This method will traverse the code, identify Script Options using defined tokens, and replace them with whitespace to ensure the cleaned script executes smoothly.

Needs: req
Covers:
- `req~script-option-removal~1`

### Java %scriptclass Option Handling in Design
`dsn~java-scriptclass-option-handling~1`

Design logic to correctly identify a single %scriptclass option within the script. Additional occurrences of %scriptclass will be removed from the script code. Use a flag to mark the first occurrence and ensure subsequent ones are discarded.

Needs: req
Covers:
- `req~java-scriptclass-option-handling~1`

### Java %jar Option Handling in Design
`dsn~java-jar-option-handling~1`

Design the parser to collect and handle multiple %jar options. Ensure that they follow the Java CLASSPATH environment variable syntax (colon-separated values), remove duplicates, and maintain the order specified in the script by using sets and lists.

Needs: req
Covers:
- `req~java-jar-option-handling~1`

### Java %jvmoption Handling in Design
`dsn~java-jvmoption-handling~1`

Create a way for the parser to collect multiple %jvmoption options, allowing duplicates and preserving the order specified. This can be achieved using a list to store the options as they are parsed.

Needs: req
Covers:
- `req~java-jvmoption-handling~1`

### Java %import Option Handling in Design
`dsn~java-import-option-handling~1`

Implement logic to process %import options by interacting with the Swig Metadata object. The parser will replace the found %import Script Option with the referenced script code and handle nested %jar, %jvmoption, and %import options, ignoring %scriptclass options in imported scripts.

Needs: req
Covers:
- `req~java-import-option-handling~1`

### General Parser Integration
`dsn~general-parser-integration~1`

Ensure that the new parser integrates seamlessly into the Exasol UDF Client environment. This includes embedding the parser within the custom C++ namespace, ensuring it meets all linker requirements, and does not introduce additional runtime dependencies.

Needs: req
Covers:
- `req~new-parser-integration~1`

## Architecture Overview

This section provides a high-level overview of the system architecture for the UDF Client Script Options parser.

### System Context

#### External Interfaces
- **UDF Client**: Communicates with the parser to process script options within UDFs.
- **Exasol Database**: Provides the data environment where UDF scripts are executed.

### Components

#### Parser Component
- Implemented using the [ctpg](https://github.com/peter-winter/ctpg) parser library with Lexer and Parser rules defined in C++.
- Manages the identification, handling, and removal of Script Options, including Java-specific options like %scriptclass, %jar, %jvmoption, and %import.

#### Integration Component
- Ensures the parser is embedded within the UDF Client namespace without runtime dependencies.
- Handles interaction with the Swig Metadata object for processing %import options.

## Detailed Design

### Parser Component Implementation

1. **Lexer Rules**
   - Define tokens for `%optionKey`, `optionValue`, and whitespace characters including `\t`, `\v`, `\f`.

2. **Parser Rules**
   - Define the grammar for identifying and processing Script Options.
   - Handle multiple options with the same key and manage duplicates as specified.
   - Implement rules to replace escape sequences (e.g., `\\`, `\n`, `\r`) in option values.
   - Ensure leading white spaces in option values are interpreted correctly.

3. **Option Handling**
   - Define a map of lists to collect and store Script Options by their keys.
   - Implement logic to detect and discard additional %scriptclass options, maintaining only the first instance.
   - Ensure %jar options follow the syntax of Java CLASSPATH and remove duplicates while preserving order using a set/list combination.

### Integration Component Implementation

1. **Namespace Embedding**
   - Ensure the parser is embedded within the custom C++ namespace of the UDF Client to meet linker requirements.

2. **Swig Metadata Interaction**
   - Implement methods to interact with the Swig Metadata object for processing %import options.
   - Replace found %import options with the referenced script code and manage any nested Script Options within the imported