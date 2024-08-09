#include "base/exaudflib/swig/script_data_transfer_objects.h"
#include <gtest/gtest.h>

using namespace ExecutionGraph;


class ImportSpecificationTest : public ::testing::Test {
protected:
    virtual void SetUp() {

    }
    virtual void TearDown() {

    }
};

TEST_F(ImportSpecificationTest, is_subselect)
{
    ImportSpecification is1(true);
    ImportSpecification is2(false);
    EXPECT_TRUE(is1.isSubselect());
    EXPECT_FALSE(is2.isSubselect());
}

TEST_F(ImportSpecificationTest, column_names_and_types_only_for_subselects)
{
    // append
    ImportSpecification is1(false);
    try {
        is1.appendSubselectColumnName("col1");
        ADD_FAILURE();
    } catch (ImportSpecification::Error& e)
    {
        EXPECT_STREQ(e.what(), "import specification error: cannot add column name to non-subselect import specification");
    }
    ImportSpecification is2(true);
    try {
        is2.appendSubselectColumnName("col1");
    } catch (ImportSpecification::Error& e)
    {
        ADD_FAILURE();
    }
    ImportSpecification is3(false);
    try {
        is3.appendSubselectColumnType("type1");
        ADD_FAILURE();
    } catch (ImportSpecification::Error& e)
    {
        EXPECT_STREQ(e.what(), "import specification error: cannot add column type to non-subselect import specification");
    }
    ImportSpecification is4(true);
    try {
        is4.appendSubselectColumnType("type1");
    } catch (ImportSpecification::Error& e)
    {
        ADD_FAILURE();
    }
    // get
    ImportSpecification is5(false);
    try {
        is5.getSubselectColumnNames();
        ADD_FAILURE();
    } catch (ImportSpecification::Error& e)
    {
        EXPECT_STREQ(e.what(), "import specification error: cannot get column names of non-subselect import specification");
    }
    ImportSpecification is6(true);
    try {
        is6.getSubselectColumnNames();
    } catch (ImportSpecification::Error& e)
    {
        ADD_FAILURE();
    }
    ImportSpecification is7(false);
    try {
        is7.getSubselectColumnTypes();
        ADD_FAILURE();
    } catch (ImportSpecification::Error& e)
    {
        EXPECT_STREQ(e.what(), "import specification error: cannot get column types of non-subselect import specification");
    }
    ImportSpecification is8(true);
    try {
        is8.getSubselectColumnTypes();
    } catch (ImportSpecification::Error& e)
    {
        ADD_FAILURE();
    }
}

TEST_F(ImportSpecificationTest, column_names_and_types)
{
    ImportSpecification is1(true);
    EXPECT_FALSE(is1.hasSubselectColumnSpecification());
    is1.appendSubselectColumnName("col1");
    EXPECT_TRUE(is1.hasSubselectColumnSpecification());
    EXPECT_FALSE(is1.hasConsistentColumns());

    is1.appendSubselectColumnType("type1");
    EXPECT_TRUE(is1.hasSubselectColumnSpecification());
    EXPECT_TRUE(is1.hasConsistentColumns());

    is1.appendSubselectColumnName("col2");
    EXPECT_TRUE(is1.hasSubselectColumnSpecification());
    EXPECT_FALSE(is1.hasConsistentColumns());

    is1.appendSubselectColumnType("type2");
    EXPECT_TRUE(is1.hasSubselectColumnSpecification());
    EXPECT_TRUE(is1.hasConsistentColumns());

    is1.appendSubselectColumnName("col3");
    EXPECT_TRUE(is1.hasSubselectColumnSpecification());
    EXPECT_FALSE(is1.hasConsistentColumns());

    is1.appendSubselectColumnType("type3");

    EXPECT_TRUE(is1.hasSubselectColumnSpecification());
    EXPECT_TRUE(is1.hasConsistentColumns());
    EXPECT_EQ(is1.getSubselectColumnNames().size(), 3);
    EXPECT_EQ(is1.getSubselectColumnTypes().size(), 3);
    EXPECT_EQ(is1.getSubselectColumnNames()[0], "col1");
    EXPECT_EQ(is1.getSubselectColumnNames()[1], "col2");
    EXPECT_EQ(is1.getSubselectColumnNames()[2], "col3");
    EXPECT_EQ(is1.getSubselectColumnTypes()[0], "type1");
    EXPECT_EQ(is1.getSubselectColumnTypes()[1], "type2");
    EXPECT_EQ(is1.getSubselectColumnTypes()[2], "type3");
}

TEST_F(ImportSpecificationTest, connection_name_or_details)
{
    ImportSpecification is1(false);
    is1.setConnectionName("some_connection");
    try {
        is1.setConnectionInformation(ConnectionInformation("some_address","some_user","some_password"));
        ADD_FAILURE();
    } catch(ImportSpecification::Error& e) {
        EXPECT_STREQ(e.what(), "import specification error: cannot set connection information, because there is already a connection name set");
    }


    //

    ImportSpecification is2(false);
    is2.setConnectionInformation(ConnectionInformation("some_address","some_user","some_password"));
    try {
        is2.setConnectionName("some_connection");
        ADD_FAILURE();
    } catch(ImportSpecification::Error& e) {
        EXPECT_STREQ(e.what(), "import specification error: cannot set connection name, because there is already connection information set");
    }
}

TEST_F(ImportSpecificationTest, connection_dont_set_twice)
{
    ImportSpecification is1(false);
    is1.setConnectionName("some_connection");
    try {
        is1.setConnectionName("some_other_connection");
        ADD_FAILURE();
    } catch(ImportSpecification::Error& e) {
        EXPECT_STREQ(e.what(), "import specification error: connection name is set more than once");
    }
    ImportSpecification is2(false);
    is2.setConnectionInformation(ConnectionInformation("some_address","some_user","some_password"));
    try {
        is2.setConnectionInformation(ConnectionInformation());
        ADD_FAILURE();
    } catch(ImportSpecification::Error& e) {
        EXPECT_STREQ(e.what(), "import specification error: cannot set connection information more than once");
    }
}


TEST_F(ImportSpecificationTest, connection_info)
{
    ImportSpecification is1(false);
    is1.setConnectionName("some_connection");
    EXPECT_EQ(is1.getConnectionName(), "some_connection");

    ImportSpecification is2(false);
    is2.setConnectionInformation(ConnectionInformation("some_address","some_user","some_password"));
    EXPECT_EQ(is2.getConnectionInformation().getAddress(), "some_address");
    EXPECT_EQ(is2.getConnectionInformation().getUser(), "some_user");
    EXPECT_EQ(is2.getConnectionInformation().getPassword(), "some_password");
    EXPECT_EQ(is2.getConnectionInformation().getKind(), "password");
}


TEST_F(ImportSpecificationTest, parameters)
{
    ImportSpecification is1(false);
    EXPECT_FALSE(is1.hasParameters());
    is1.addParameter("key1","value1");
    EXPECT_TRUE(is1.hasParameters());
    std::map<std::string, std::string>::const_iterator ps1 = is1.getParameters().begin();
    EXPECT_EQ(ps1->first, "key1");
    EXPECT_EQ(ps1->second, "value1");

    is1.addParameter("key2","value2");
    EXPECT_TRUE(is1.hasParameters());
    std::map<std::string, std::string>::const_iterator ps2 = is1.getParameters().begin();
    ++ps2;
    EXPECT_EQ(ps2->first, "key2");
    EXPECT_EQ(ps2->second, "value2");
}

//
//
//
//
class ConnectionInformationTest : public ::testing::Test {
protected:
    virtual void SetUp() {

    }
    virtual void TearDown() {

    }
};

TEST_F(ConnectionInformationTest, address_user_password_construction)
{
    ConnectionInformation ci("some_address","some_user","some_password");
    EXPECT_EQ(ci.getKind(), "password");
    EXPECT_EQ(ci.getAddress(), "some_address");
    EXPECT_EQ(ci.getUser(), "some_user");
    EXPECT_EQ(ci.getPassword(), "some_password");
    EXPECT_FALSE(ci.hasData());
}

TEST_F(ConnectionInformationTest, kind_address_user_password_construction)
{
    ConnectionInformation ci("some_kind","some_address","some_user","some_password");
    EXPECT_EQ(ci.getKind(), "some_kind");
    EXPECT_EQ(ci.getAddress(), "some_address");
    EXPECT_EQ(ci.getUser(), "some_user");
    EXPECT_EQ(ci.getPassword(), "some_password");
    EXPECT_FALSE(ci.hasData());
}

TEST_F(ConnectionInformationTest, empty_construction)
{
    ConnectionInformation ci;
    EXPECT_TRUE(ci.getKind().empty());
    EXPECT_TRUE(ci.getAddress().empty());
    EXPECT_TRUE(ci.getUser().empty());
    EXPECT_TRUE(ci.getPassword().empty());
    EXPECT_TRUE(ci.hasData());
}

TEST_F(ConnectionInformationTest, copy)
{
    ConnectionInformation ca("a","b","c","d");
    ConnectionInformation cb(ca);
    EXPECT_EQ(ca.getKind(), cb.getKind());
    EXPECT_EQ(ca.getAddress(), cb.getAddress());
    EXPECT_EQ(ca.getUser(), cb.getUser());
    EXPECT_EQ(ca.getPassword(), cb.getPassword());
    EXPECT_EQ(ca.hasData(), cb.hasData());
}
