#!/bin/sh


all: pythonclient.tar.gz

script_client.proto:
	wget https://raw.githubusercontent.com/EXASOL/script-languages/master/src/zmqcontainer.proto

script_client_pb2.py: script_client.proto
	protoc -I. $<  --python_out=.


pythonclient.tar.gz: script_client_pb2.py common_client.py
	tar -zcf pythonclient.tar.gz common_client.py python2 python3 script_client_pb2.py
	
.PHONY: clean
clean:
	@rm -f script_client_pb2.py pythonclient.tar.gz

