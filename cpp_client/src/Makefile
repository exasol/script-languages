#!/bin/sh

CXXFLAGS=-O3 -I. -Wall -Werror -fPIC -pthread -DNDEBUG -std=c++11

LDFLAGS=-rdynamic -lzmq -lprotobuf -lpthread -lcrypto -ldl

all: cppclient.tar.gz


jsoncpp-master:
	wget -N https://github.com/open-source-parsers/jsoncpp/archive/master.zip -O jsoncpp-master.zip
	unzip jsoncpp-master.zip
	cd jsoncpp-master && python amalgamate.py

script_client.proto:
	wget https://github.com/EXASOL/script-languages/blob/master/script_client.proto

script_client.pb.cc: script_client.proto
	protoc -I. $<  --cpp_out=.


%.o: %.cc script_client.pb.cc wrapper.h
	g++ -c $(CXXFLAGS) -o $@ $<

%.o: %.c
	g++ -c $(CXXFLAGS) -o $@ $<


cppclient: jsoncpp-master main.o script_client.pb.o jsoncpp-master/dist/jsoncpp.o wrapper.h
	g++ -o cppclient main.o script_client.pb.o jsoncpp-master/dist/jsoncpp.o  $(LDFLAGS)

cppclient.tar.gz: cppclient jsoncpp-master
	tar --transform 's/.*\///g' -zcf cppclient.tar.gz cppclient wrapper.h script_client.pb.h jsoncpp-master/dist/json/json.h jsoncpp-master/dist/json/json-forwards.h

.PHONY: clean
clean:
	@rm -f *.o
	@rm -f cppclient cppclient.tar.gz script_client.pb.cc script_client.pb.h
	@rm -rf jsoncpp-master jsoncpp-master.zip
