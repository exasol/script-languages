#!/usr/opt/bs-python-2.7/bin/python
# encoding: utf8

import sys
# Temporary!
sys.path.append("/x/u/zg1089/hadoop-etl/python-kerberos/kerberos-1.2.2/build/lib.linux-x86_64-2.7")
import kerberos

class GssClient:
    def __init__(self, service, principal):
        self.service = service
        self.principal = principal
        self.ctx = None

    def get_auth_header(self):
        if self.ctx:
            raise RuntimeError("Context has already been initialized")
        __, self.ctx = kerberos.authGSSClientInit(self.service, principal = self.principal)
        kerberos.authGSSClientStep(self.ctx, "")
        token = kerberos.authGSSClientResponse(self.ctx)
        return {"Authorization": "Negotiate " + token}

    def check_auth_header(self, auth_header):
        if not self.ctx:
            raise RuntimeError("Invalid context: " + self.ctx)
        if not auth_header:
            raise RuntimeError("www-authenticate header is not valid: " + auth_header)
        auth_val = auth_header.split(" ", 1)
        if len(auth_val) != 2 or auth_val[0].lower() != "negotiate":
            raise RuntimeError("www-authenticate header is not valid: " + auth_header)
        kerberos.authGSSClientStep(self.ctx, auth_val[1])
        kerberos.authGSSClientClean(self.ctx)
        self.ctx = None
