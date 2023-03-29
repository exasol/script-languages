CREATE JAVA SCALAR SCRIPT vectorsize5000(A DOUBLE)
RETURNS VARCHAR(2000000) AS
class VECTORSIZE5000 {
    static String numbers;

    static {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 5000; i++) {
            sb.append(Integer.toString(i));
        }
        numbers = sb.toString();
    }

    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        return numbers;
    }
}
/

CREATE JAVA SCALAR SCRIPT
vectorsize(length DOUBLE, dummy DOUBLE)
RETURNS VARCHAR(2000000) AS
import java.util.Map;
import java.util.HashMap;

class VECTORSIZE {
    static Map<Integer, String> cache = new HashMap<Integer, String>();

    static void fillCache(int len) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len; i++) {
            sb.append(Integer.toString(i));
        }
        cache.put(len, sb.toString());
    }

    static String run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        int len = ctx.getInteger("length");
        if (!cache.containsKey(len))
            fillCache(len);
        return cache.get(len);
    }
}
/

CREATE JAVA SCALAR SCRIPT
vectorsize_set(length DOUBLE, n DOUBLE, dummy DOUBLE)
EMITS (o VARCHAR(2000000)) AS
import java.util.Map;
import java.util.HashMap;

class VECTORSIZE_SET {
    static Map<Integer, String> cache = new HashMap<Integer, String>();

    static void fillCache(int len) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len; i++) {
            sb.append(Integer.toString(i));
        }
        cache.put(len, sb.toString());
    }

    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        int len = ctx.getInteger("length");
        if (!cache.containsKey(len))
            fillCache(len);
        int n = ctx.getInteger("n");
        for (int i = 0; i < n; i++) {
            ctx.emit(cache.get(len));
        }
    }
}
/

-- vim: ts=2:sts=2:sw=2:et:fdm=indent
