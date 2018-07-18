#!/usr/opt/bs-python-2.7/bin/python

import os
import sys

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class ComplexQueries(udf.TestCase):
    def setUp(self):
        self.query('DROP SCHEMA FN2 CASCADE', ignore_errors=True)
        self.query('CREATE SCHEMA FN2')


    def test_udf_merge(self):
        self.query(udf.fixindent('''
        create or replace lua scalar script
        my_range("from" number, "to" number)
        emits(x number) as
        function run(ctx)
            for i = ctx.from, ctx.to do
                ctx.emit(i)
            end
        end'''))
        self.query(udf.fixindent('''
        create or replace lua scalar script
        my_sqrt(x number)
        returns number as
        function run(ctx)
            return math.sqrt(ctx.x)
        end'''))
        self.query('''create or replace table ttt1 as (select my_range(1,1000000) from dual);''')
        self.query('''create or replace table ttt2 as select * from (select x,x*x from ttt1) as t(id,x)''')
        self.query('''create or replace table ttt3 as select * from (select x,x*x*x from ttt1 where mod(x,2) = 0) as t(id,x)''')
        self.query('''merge into ttt2 t2 using ttt3 t3 on (t2.id = t3.id) when matched then update set t2.x = my_sqrt(t3.x) where t2.id = t3.id''')
        self.assertEqual(1, self.query('''select floor(sum(x)*10e-18) from ttt2''')[0][0])

    def test_interleaving_sleeps(self):
        self.query(udf.fixindent('''
        create or replace lua scalar script
        wait_n_echo(v number, t number)
        returns number as
        local socket = require "socket"
        function run (ctx)
            socket.sleep(ctx.t)
            return math.random(ctx.v)
        end
        '''))
        rows = self.query('''select sum(s) from (select sum(wait_n_echo(1,x)) s from (values 1,2,3) as t(x) group by x)''')
        self.assertEqual(3,rows[0][0])

    def test_join_emits_on_scalar_udf(self):
        self.query(udf.fixindent('''
                CREATE or replace lua SCALAR SCRIPT
                up_or_down (s number, up boolean)
                emits(x double) AS
                function run(context)
                    if context.up then
                        for i = 1,context.s,1 do
                            context.emit(i)
                        end
                    else
                        for i = context.s,1,-1 do
                            context.emit(i)
                        end
                    end
                end
                '''))
        self.query(udf.fixindent('''
                create or replace lua scalar script
                lleql(x number, y number)
                returns boolean AS
                function run(ctx)
                    return ctx.x * 2 == ctx.y
                end
        '''))
        rows = self.query('''
        select sum(u+v) as su1
        from (select fn2.up_or_down(x+y, false)
              from (select sum(u) as su
                    from (select (y-x) u
                          from (select *
                                from (select fn2.up_or_down(5000,false)
                                      from dual)
                                group by x) as t1(x)
                                ,
                               (select fn2.up_or_down(3,true)
                                from dual) t2(y)
                          where fn2.lleql(x,y))) as t1(x)
                   ,   
                   (select sum(u) as su
                    from (select (y-x) u
                          from (select *
                                from (select fn2.up_or_down(71,false)
                                      from dual)
                                group by x) as t1(x)
                               ,
                               (select fn2.up_or_down(10000,true)
                                from dual) t2(y)
                          where fn2.lleql(x,y))) as t2(y)) as t11(u)
             ,
             (select fn2.up_or_down(x+y, true)
              from (select sum(u) as su
                    from (select (y-x) u
                          from (select *
                                from (select fn2.up_or_down(500,false)
                                      from dual)
                                group by x) as t1(x)
                                ,
                                (select *
                                 from (select fn2.up_or_down(1500,true)
                                       from dual)
                                 group by x) as t2(y)
                          where fn2.lleql(x,y))) as t1(x)
                    ,
                    (select sum(u) as su
                     from (select (y-x) u
                           from (select fn2.up_or_down(5,false)
                                 from dual) t1(x)
                                ,
                                (select fn2.up_or_down(5,true)
                                 from dual) t2(y)
                           where fn2.lleql(x,y))) as t2(y)) as t21(v)
        where ( fn2.lleql(u,v) or 1 = 1)
        ''')
        self.assertEqual(20467297383426, rows[0][0])

        


if __name__ == '__main__':
    udf.main()

# vim: ts=4:sts=4:sw=4:et:fdm=indent

