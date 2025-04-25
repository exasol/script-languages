--CREATE <lang>  SCALAR SCRIPT
--base_pi()
--RETURNS DOUBLE AS

-- pi

CREATE OR REPLACE java SCALAR SCRIPT
DEFAULT_TZ()
RETURNS VARCHAR(100) AS
import java.time.ZoneId;
import java.util.TimeZone;

class DEFAULT_TZ{
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        TimeZone timeZone = TimeZone.getDefault();
        return timeZone.getDisplayName(false, TimeZone.SHORT);

    }
}
/

CREATE OR REPLACE java SCALAR SCRIPT
MODIFY_TZ_TO_NEW_YORK()
RETURNS VARCHAR(100) AS
%jvmoption -Duser.timezone=America/New_York;
import java.time.ZoneId;
import java.util.TimeZone;

class MODIFIED_TZ{
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        TimeZone timeZone = TimeZone.getDefault();
        return timeZone.getDisplayName(false, TimeZone.SHORT);
    }
}
/