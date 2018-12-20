# Exasol script language flavors overview
Script language flavors are configurations of languages and libraries to be available in script languages for the Exasol database.
Currently, we have
* the `standard` flavors: These flavors include three langauage implementations: Java, Python, and R and a collection of typical libraries
* the `python3-ds` flavors: These flavors include Python3 as language and a number of typical data science and machine learning libraries
* the `fancyr` flavors: These flavors include R as language and a large collection of popular R packages.

# Naming convention
Script language flavors for different purposes come in different versions that are reflected by their name.
The naming convention is like this:

`<flavor-name>-EXASOL-<minimum-Exasol-Version>`

This allows us to provide new versions of a flavor when new features become available in Exasol that are not supported in older flavors.
For instance, the flavors for Exasol 6.1.0 support newer Linux distributions as their basis and hence, overall contain newer versions of libraries and languages.

So, in order to find the correct version of a flavor for your version of Exasol, simply take the latest container with version less than or equal to your Exasol versio.
So for Exasol 6.1.1, you would use the `*-EXASOL-6.1.0` flavors, while for Exasol 6.0.14 you would use the `*-EXASOL-6.0.0` flavors.
