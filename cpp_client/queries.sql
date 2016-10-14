create schema cpp;
open schema cpp;
CREATE OR REPLACE PYTHON SCALAR SCRIPT sh(cmd VARCHAR(2000000)) EMITS (line VARCHAR(2000000)) AS
import subprocess

def run(c):
    try:
        p = subprocess.Popen(c.cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, close_fds = True, shell = True)
        out, err = p.communicate()
        for line in out.split('\n'):
            c.emit(line)
    finally:
        if p is not None:
            try: p.kill()
            except: pass
/


select sh('ls -l /');

select sh('ls -l /buckets/bfsdefault/cpp/tiny-dnn-master/');

open schema cpp;
alter session set script_languages = 'PYTHON=builtin_python R=builtin_r JAVA=builtin_java CPP=localzmq+protobuf:///bfsdefault/default/EXAClusterOS/ScriptLanguages-6.0.0#buckets/bfsdefault/cpp/cppclient/cppclient';



CREATE cpp SCALAR SCRIPT scaleAndDuplicate(n INT, x DOUBLE, y DOUBLE, z DOUBLE)
EMITS (x double, y double, z double) AS

%compilerflags -lblas;
#include <cblas.h>

using namespace UDFClient;

void run_cpp(Metadata* meta, InputTable* in, OutputTable* out) {
  for (size_t n=0; n < in->getInteger(0); ++n) {
    double x[] = {in->getDouble(1), in->getDouble(2), in->getDouble(3)};
    cblas_dscal(3, 4.323, x, 1);
    for (size_t i = 0; i < 3; ++i)
        out->setDouble(i,x[i]);
    in->next();
    out->next();
  }
}
/

create or replace python scalar script x(y int) returns int as
def xx:a
/

select x(12);


create or replace cpp scalar script tc(x double) returns double as

using namespace UDFClient;

#include <cmath>

void run_cpp(Metadata* meta, InputTable* iter, OutputTable* res)
{
        //iter->reset();
        //iter->next();
        res->setDouble(0,sin(iter->getDouble(0)));
        res->next();
        res->flush();
}

/




select tc(32);



create or replace cpp scalar script scale(x double, y double, z double) emits (x double, y double, z double) as
%compilerflags -lblas;
#include <cblas.h>

using namespace UDFClient;

void run_cpp(Metadata* meta, InputTable* iter, OutputTable* res)
{

  double x[] = { iter->getDouble(0), iter->getDouble(1), iter->getDouble(2) };
  cblas_dscal(3, 4.323, x, 1);
 
  for (int i = 0; i < 3; ++i)
      res->setDouble(i,x[i]);
  res->next();
  res->flush();
}
/

select scale(1,2,3);



create or replace cpp scalar script importMe() returns int as

 void scaleIt(const int n, const double a, double *x, const int incx)
 {
       cblas_dscal (n, a, x, incx);
 }
/


create or replace cpp scalar script scale(x double, y double, z double) emits (x double, y double, z double) as
%compilerflags -lblas;

#include <cblas.h>

%import importMe;

using namespace UDFClient;

void run_cpp(Metadata* meta, InputTable* iter, OutputTable* res)
{

  double x[] = { iter->getDouble(0), iter->getDouble(1), iter->getDouble(2) };
  scaleIt(3, 4.323, x, 1);
 
  for (int i = 0; i < 3; ++i)
      res->setDouble(i,x[i]);
  res->next();
  res->flush();
}


/


select scale(1,2,3);



create or replace cpp scalar script se() emits (...) as
#include <string>
#include "cJSON.h"

using namespace UDFClient;

void run_cpp(Metadata* meta, InputTable* iter, OutputTable* res)
{
  res->setDouble(0,10);
  res->next();
  res->flush();
}


std::string getDefaultOutputColumns(Metadata* meta)
{
   return "z double";
}

std::string adapterCall(Metadata* meta, const std::string input) {

}

/



create or replace cpp adapter script cas as
{}
;

/


create or replace cpp scalar script cas2() returns int as

#include <string>
#include "json.h"

using namespace UDFClient, std, json;

string adapterCall(Metadata* meta, const string input) {
    Value root(input);
    string res;

    if (root["type"] == "createVirtualSchema") {
        return "{\"type\": \"createVirtualSchema\",\"schemaMetadata\": {{\"tables\": [{{\"name\": \"DUMMY\",\"columns\": [{{\"name\": \"KEY\",
   \"dataType\": {{\"type\": \"VARCHAR\", \"size\": 2000000}}}},{{\"name\": \"VALUE\",\"dataType\": {{\"type\": \"VARCHAR\", \"size\": 2000000}}}}]}}]}}}";
    } else if (root["type"] == "dropVirtualSchema") {
         return "{\"type\": \"dropVirtualSchema\"}";
    } else if (root["type"] == "setProperties") {
         return "{\"type\": \"setProperties\"}";
    } else if (root["type"] == "refresh") {
         return "{\"type\": \"refresh\"}";
    } else if (root["type"] == "getCapabilities") {
         return "{\"type\": \"getCapabilities\","capabilities": []}";
    } else if (root["type"] == "pushdown") {
        return = "{\"type\": \"pushdown\", \"sql\": \"SELECT * FROM (VALUES ('FOO', 'BAR')) t\"}";
   } else {
     throw LanguagePlugin::exception("Unsupported callback")
   }

}

/





create or replace cpp adapter script cas as
%compilerflags -std=c++11;
#include <string>
#include "json.h"
#include <memory>

using namespace UDFClient;
using namespace std;

Json::Value parseJson(const string& json)
{
    Json::Value root;
    Json::CharReaderBuilder readerBuilder;
    readerBuilder["collectComments"] = false;
    shared_ptr<Json::CharReader> reader(readerBuilder.newCharReader());
    std::string errs;
    bool ok = reader->parse(json.c_str(), json.c_str() + json.size(),& root, & errs);
    if (!ok) {
        throw LanguagePlugin::exception(errs.c_str());
    }
    return root;
}

string adapterCall2(Metadata* meta, const string input) {
    Json :: Value root = parseJson(input);
    string res;

	cerr << "ADAPTER CALL: " << input << endl;


    if (root["type"] == "createVirtualSchema") {
        return "{\"type\": \"createVirtualSchema\", \"schemaMetadata\": {\"tables\": [{\"name\": \"DUMMY\", \"columns\": [{\"name\": \"KEY\",\"dataType\": {\"type\": \"VARCHAR\", \"size\": 2000000}}]}]}}";
    } else if (root["type"] == "dropVirtualSchema") {
         return "{\"type\": \"dropVirtualSchema\"}";
    } else if (root["type"] == "setProperties") {
         return "{\"type\": \"setProperties\"}";
    } else if (root["type"] == "refresh") {
         return "{\"type\": \"refresh\"}";
    } else if (root["type"] == "getCapabilities") {
         return "{\"type\": \"getCapabilities\",\"capabilities\": []}";
    } else if (root["type"] == "pushdown") {
        return "{\"type\": \"pushdown\", \"sql\": \"SELECT * FROM (VALUES 'FOO', 'BAR')\"}";
   } else {
     throw LanguagePlugin::exception("Unsupported callback");
   }

}

string adapterCall(Metadata* meta, const string input) {
    string res = adapterCall2(meta,input);
    cerr << "Result = " << res;
    return res;
}
/


open schema cpp;
drop virtual schema vs1 cascade;
create virtual schema vs1 using cpp.cas;

select * from vs1.dummy;



--
--

create or replace cpp scalar script import_from_me() emits (x double) as
#include <string>

using namespace UDFClient;

void run_cpp(Metadata* meta, InputTable* iter, OutputTable* res)
{
  for (size_t i=0; i<100; i++)
  {
    res->setDouble(0,i);
    res->next();      
  }
  //res->flush();
}



std::string generateSqlForImportSpec(Metadata* meta, const ImportSpecification& importSpecification) {
   return "select 1,2,3";
}




/

select import_from_me();


select * from (import from script import_from_me);





create or replace cpp scalar script con() emits (x varchar(10000)) as

void run_cpp(UDFClient::Metadata* meta, UDFClient::InputTable* in, UDFClient::OutputTable* out)
{
	UDFClient::ConnectionInformation con = meta->connectionInformation("asdsa");
	std::string s = con.getKind()+"--"+con.getAddress()+"---"+con.getUser()+"---"+con.getPassword();	
    out->setString(0, s.c_str(), s.size());
	out->next();
	out->next();
	out->next();
	out->setString(0, s.c_str(), s.size());
	out->next();
	out->next();
}

/




create or replace cpp scalar script all_of_it() returns double as

using namespace UDFClient;

void run_cpp(const Metadata& meta, InputTable& in, OutputTable& out)
{
	out.setDouble(0,12.5);
	out.next();
}

std::string generateSqlForImportSpec(const Metadata& meta, const ImportSpecification& importSpecification) {
   return "select 1,2,3";
}

std::string adapterCall(const Metadata& meta, const std::string input) {
    string res = adapterCall2(meta,input);
    cerr << "Result = " << res;
    return res;
}

std::string getDefaultOutputColumns(const Metadata& meta)
{
   return "z double";
}

/


select all_of_it();

create or replace connection asdsa to 'HOST' USER 'me' identified by 'you';

select con();

select * from exa_parameters where parameter_name='TIME_ZONE_BEHAVIOR';

select sh('find /buckets 2>& 1 | grep exaudfclient');
select sh('ls /buckets/bfsdefault/default/EXAClusterOS');
select * from exa_parameters;
select param_value as 'val' from exa_metadata where param_name='databaseProductVersion';