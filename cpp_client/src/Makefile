#!/bin/sh

CXXFLAGS=-O3 -I. -Wall -fPIC -pthread -DNDEBUG -std=c++11 -DENABLE_CPP_VM -DEXTERNAL_PROCESS -DBUILDINSWIGDIR

LDFLAGS=-rdynamic -lzmq -lprotobuf -lpthread -lcrypto -ldl

all: cppclient.tar.gz


jsoncpp-master.zip:
	wget -N https://github.com/open-source-parsers/jsoncpp/archive/master.zip -O jsoncpp-master.zip

jsoncpp-master: jsoncpp-master.zip
	unzip jsoncpp-master.zip
	cd jsoncpp-master && python amalgamate.py

zmqcontainer.pb.cc: zmqcontainer.proto
	        protoc -I. $<  --cpp_out=.

swigcontainers_ext.h: swigcontainers.h
	cpp -DEXTERNAL_PROCESS swigcontainers.h | sed 's/^\$$/#/; /^# *[0-9] */d; /^ *$$/d' > "swigcontainers_ext.h"


%.o: %.cc zmqcontainer.pb.cc swigcontainers_ext.h scriptDTOWrapper.h scriptDTO.h
	g++ -c $(CXXFLAGS) -o $@ $< -DBUILDINSWIGDIR

%.o: %.c
	g++ -c $(CXXFLAGS) -o $@ $<


cppclient: jsoncpp-master zmqcontainerclient.o zmqcontainer.pb.o jsoncpp-master/dist/jsoncpp.o swigcontainers_ext.h scriptDTOWrapper.h scriptDTO.o scriptDTOWrapper.o cpp.o
	g++ -o cppclient zmqcontainerclient.o zmqcontainer.pb.o jsoncpp-master/dist/jsoncpp.o scriptDTO.o scriptDTOWrapper.o cpp.o $(LDFLAGS)

cppclient.tar.gz: cppclient jsoncpp-master
	tar --transform 's/.*\///g' -zcf cppclient.tar.gz cppclient swigcontainers_ext.h scriptDTOWrapper.h scriptDTO.h zmqcontainer.pb.h jsoncpp-master/dist/json/json.h jsoncpp-master/dist/json/json-forwards.h

.PHONY: clean
clean:
	@rm -f *.o *~
	@rm -f cppclient cppclient.tar.gz zmqcontainer.pb.cc zmqcontainer.pb.h swigcontainers_ext.h
	@rm -rf jsoncpp-master jsoncpp-master.zip
