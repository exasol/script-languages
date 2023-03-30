create java scalar script dob_1i_1o(x double) emits(y double)
as
class DOB_1I_1O{
        static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x"));
        ctx.emit(ctx.getDouble("x"));
        }
}
/
create java scalar script line_1i_1o(x double) emits(y double)
as
class LINE_1I_1O{
        static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x"));
        }
}
/
create java scalar script line_1i_2o(x double) emits(y double, z double)
as
class LINE_1I_2O{
        static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x"),ctx.getDouble("x"));
        }
}
/
create java scalar script line_2i_1o(x double, y double) emits(z double)
as
class LINE_2I_1O{
        static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x")+ctx.getDouble("y"));
        }
}
/
create java scalar script line_3i_2o(x double, y double, z double) emits(z1 double, z2 double)
as
class LINE_3I_2O{
        static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
                ctx.emit(ctx.getDouble("x")+ctx.getDouble("y"),3000);
        }
}
/
