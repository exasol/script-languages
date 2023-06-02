CREATE java SCALAR SCRIPT
performance_map_words(w VARCHAR(1000))
EMITS (w VARCHAR(1000), c INTEGER) AS
import java.util.regex.Pattern;
import java.util.regex.Matcher;
class PERFORMANCE_MAP_WORDS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String w = ctx.getString("w");
        if (w != null) {
            Matcher m = Pattern.compile("([\\p{Alnum}\\p{Punct}]+)").matcher(w);
            while (m.find()) {
                ctx.emit(m.group(), 1);
            }
        }
    }
}
/

CREATE java SCALAR SCRIPT
performance_map_unicode_words(w VARCHAR(1000))
EMITS (w VARCHAR(1000), c INTEGER) AS
import java.util.regex.Pattern;
import java.util.regex.Matcher;
class PERFORMANCE_MAP_UNICODE_WORDS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String w = ctx.getString("w");
        if (w != null) {
            Matcher m = Pattern.compile("([\\p{Alnum}\\p{Punct}]+)", Pattern.UNICODE_CHARACTER_CLASS).matcher(w);
            while (m.find()) {
                ctx.emit(m.group(), 1);
            }
        }
    }
}
/

CREATE java SET SCRIPT
performance_reduce_counts(w VARCHAR(1000), c INTEGER)
EMITS (w VARCHAR(1000), c INTEGER) AS
class PERFORMANCE_REDUCE_COUNTS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String word = ctx.getString("w");
        int count = 0;
        do {
            count += ctx.getInteger("c");
        } while (ctx.next());
        ctx.emit(word, count);
    }
}
/

CREATE java SCALAR SCRIPT
performance_map_characters(text VARCHAR(1000))
EMITS (w CHAR(1), c INTEGER) AS
class PERFORMANCE_MAP_CHARACTERS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        String text = ctx.getString("text");
        if (text != null) {
            for (int i = 0; i < text.length(); i++) {
                if (Character.isHighSurrogate(text.charAt(i))) {
                    ctx.emit(text.substring(i, i + 2), 1);
                    i++;
                }
                else {
                    ctx.emit(text.substring(i, i + 1), 1);
                }
            }
        }
    }
}
/

CREATE java SET SCRIPT
performance_reduce_characters(w CHAR(1), c INTEGER)
EMITS (w CHAR(1), c INTEGER) AS
class PERFORMANCE_REDUCE_CHARACTERS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        int c = 0;
        String w = ctx.getString("w");
        if (w != null) {
            do {
                c += 1;
            } while (ctx.next());
            ctx.emit(w, c);
        }
    }
}
/

-- vim: ts=2:sts=2:sw=2
