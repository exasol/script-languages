import json
import string

def adapter_call(request):
    root = json.loads(request)
    if root["type"] == "createVirtualSchema":
        return handleCreateVSchema(root)
    elif root["type"] == "dropVirtualSchema":
        return json.dumps({"type": "dropVirtualSchema"}).encode('utf-8')
    elif root["type"] == "refresh":
        return json.dumps({"type": "refresh"}).encode('utf-8')
    elif root["type"] == "setProperties":
        return json.dumps({"type": "setProperties"}).encode('utf-8')
    if root["type"] == "getCapabilities":
        return json.dumps({
            "type": "getCapabilities",
            "capabilities": []
            }).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
    elif root["type"] == "pushdown":
        return handlePushdown(root)
    else:
        raise ValueError('Unsupported callback')

def handleCreateVSchema(root):
    res = {
        "type": "createVirtualSchema",
        "schemaMetadata": {
            "tables": [
            {
                "name": "table_1",
                "columns": [{
                    "name": "column_1",
                    "dataType": {"type": "VARCHAR", "size": 2000000}
                },{
                    "name": "column_2",
                    "dataType": {"type": "VARCHAR", "size": 2000000}
                }]
            }]
        }
    }
    return json.dumps(res).encode('utf-8')

def handlePushdown(root):
    res = {
        "type": "pushdown",
        "sql": "select dummy from dual"
        }
    return json.dumps(res).encode('utf-8')
