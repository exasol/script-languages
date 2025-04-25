create python3 SCALAR SCRIPT
DEFAULT_TZ()
RETURNS VARCHAR(100) AS
import time
def run(ctx):
    return time.tzname[0]
/

create python3 SCALAR SCRIPT
MODIFY_TZ_TO_NEW_YORK()
RETURNS VARCHAR(100) AS
import time
import os
def run(ctx):
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    return time.tzname[0]
/

-- vim: ts=4:sts=4:sw=4
