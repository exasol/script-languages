#!/usr/bin/env python3

from exasol_python_test_framework import udf
from exasol_python_test_framework.udf import timer, skip


class _JavaUdfSetup(udf.TestCase):

    def _setup_fn1_schema(self):
        self.query('DROP SCHEMA FN1 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN1')
        self.query('OPEN SCHEMA FN1')

    def _create_wordcount_udfs(self):
        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT PERFORMANCE_MAP_WORDS(w VARCHAR(1000))
            EMITS (w VARCHAR(1000), c INTEGER) AS
            class PERFORMANCE_MAP_WORDS {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String text = ctx.getString("w");
                    if (text != null) {
                        String[] words = text.split("[^\\\\w]+");
                        for (String word : words) {
                            if (!word.isEmpty()) {
                                ctx.emit(word, 1);
                            }
                        }
                    }
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT PERFORMANCE_REDUCE_COUNTS(w VARCHAR(1000), c INTEGER)
            EMITS (w VARCHAR(1000), c INTEGER) AS
            class PERFORMANCE_REDUCE_COUNTS {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String word = ctx.getString("w");
                    int count = 0;
                    do {
                        if (ctx.getInteger("c") != null) {
                            count += ctx.getInteger("c");
                        }
                    } while (ctx.next());
                    ctx.emit(word, count);
                }
            }
            /
        '''))

    def _create_character_udfs(self):
        self.query(udf.fixindent('''
            CREATE JAVA SCALAR SCRIPT PERFORMANCE_MAP_CHARACTERS(text VARCHAR(1000))
            EMITS (w CHAR(1), c INTEGER) AS
            class PERFORMANCE_MAP_CHARACTERS {
                static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                    String text = ctx.getString("text");
                    if (text != null) {
                        for (int i = 0; i < text.length(); i++) {
                            if (Character.isHighSurrogate(text.charAt(i))) {
                                ctx.emit(text.substring(i, i + 2), 1);
                                i++;
                            } else {
                                ctx.emit(text.substring(i, i + 1), 1);
                            }
                        }
                    }
                }
            }
            /
        '''))

        self.query(udf.fixindent('''
            CREATE JAVA SET SCRIPT PERFORMANCE_REDUCE_CHARACTERS(w CHAR(1), c INTEGER)
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
        '''))


class WordCount(_JavaUdfSetup):

    def setUp(self):
        self._setup_fn1_schema()
        self._create_wordcount_udfs()

    def test_word_count(self):
        sql = '''
        SELECT COUNT(*) FROM (
            SELECT fn1.performance_reduce_counts(w, c)
            FROM (
                SELECT fn1.performance_map_words(varchar02)
                FROM test.enginetablebig1
            )
            GROUP BY w
            ORDER BY 1 DESC)'''

        with timer() as t:
            ret = self.query(sql)
        print("test_word_count query:", t.duration, repr(ret))
        self.assertLessEqual(t.duration, 160)


@skip('csv data for tables wiki_freq and wiki_names is currently not available')
class FrequencyAnalysis(_JavaUdfSetup):
    maxDiff = 1024 * 20

    def setUp(self):
        self._setup_fn1_schema()
        self._create_character_udfs()

    def compare(self, old, new):
        self.log.info('compare new data with reference data')
        n_old = len(list(old))
        n_new = len(list(new))
        self.log.info('old data has %d lines', n_old)
        self.log.info('new data has %d lines', n_new)
        if max(n_old, n_new) <= 50:
            self.assertEqual(old, new)
        else:
            self.log.info('switching to compact comparison')
            old_set = set(old)
            new_set = set(new)
            only_new = list(sorted(new_set.difference(old_set)))
            only_old = list(sorted(old_set.difference(new_set)))
            if max(len(only_new), len(only_old)) <= 200:
                self.assertEqual(([], []), (only_old, only_new))
            else:
                self.log.info('diff is still too big')
                self.fail("difference: +%d/-%d elements" %
                          (len(only_new), len(only_old)))


if __name__ == '__main__':
    udf.main()
