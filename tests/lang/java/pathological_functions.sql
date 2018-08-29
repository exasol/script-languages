CREATE java SCALAR SCRIPT
sleep("sec" double)
RETURNS double AS
class SLEEP {
    static double run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        int sec = ctx.getInteger("sec");
        Thread.sleep(sec * 1000);
        return sec;
    }
}
/

CREATE java SCALAR SCRIPT
mem_hog("mb" int)
RETURNS int AS
import java.util.Vector;

class MEM_HOG {
    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        byte[] a = new byte[1024 * 1024];
        for (int i = 0; i < a.length; i++)
            a[i] = 'x';
        String str = new String(a);
        int mb = ctx.getInteger("mb");
        Vector<String> b = new Vector<String>();
        for (int i = 0; i < mb; i++)
            b.addElement(str + Integer.toString(i));
        return b.size();
    }
}
/

CREATE java SCALAR SCRIPT
cleanup_check("raise_exc" boolean, "sleep" int)
RETURNS int AS
class CLEANUP_CHECK {
    static int sleep = 0;

    static int run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        sleep = ctx.getInteger("sleep");
        if (ctx.getBoolean("raise_exc") == true)
            throw new RuntimeException();
        return 42;
    }

    static void cleanup(ExaMetadata exa) {
        Thread.sleep(sleep);
    }
}
/

-- vim: ts=4:sts=4:sw=4:et:fdm=indent
