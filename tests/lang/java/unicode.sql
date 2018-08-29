CREATE java SCALAR SCRIPT
unicode_len(word VARCHAR(1000))
RETURNS INT AS
class UNICODE_LEN {
    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String s = ctx.getString("word");
        return (s == null) ? 0 : s.codePointCount(0, s.length());
    }
}
/

CREATE java SCALAR SCRIPT
unicode_upper(word VARCHAR(1000))
RETURNS VARCHAR(1000) AS
class UNICODE_UPPER {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String s = ctx.getString("word");
        return (s == null) ? null : s.toUpperCase();
    }
}
/

CREATE java SCALAR SCRIPT
unicode_lower(word VARCHAR(1000))
RETURNS VARCHAR(1000) AS
class UNICODE_LOWER {
    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String s = ctx.getString("word");
        return (s == null) ? null : s.toLowerCase();
    }
}
/

CREATE java SCALAR SCRIPT
unicode_count(word VARCHAR(1000), convert_ INT)
EMITS (uchar VARCHAR(1), count INT) AS
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.Iterator;
class UNICODE_COUNT {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String s = ctx.getString("word");
        if (s == null)
            return;
        if (ctx.getInteger("convert_") > 0)
            s = s.toUpperCase();
        else if (ctx.getInteger("convert_") < 0)
            s = s.toLowerCase();

        Map<Integer, Integer> count = new HashMap<Integer, Integer>();
        for (int i = 0; i < s.length(); i++) {
            Integer codepoint = s.codePointAt(i);
            Integer num = count.get(codepoint);
            count.put(codepoint, (num == null ? 1 : num + 1));
            if (Character.isHighSurrogate(s.charAt(i)))
                i++;
        }
        Iterator<Map.Entry<Integer, Integer>> iter = count.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry<Integer, Integer> item = iter.next();
            ctx.emit(new String(Character.toChars(item.getKey())), item.getValue());
        }
    }
}
/

-- vim: ts=2:sts=2:sw=2
