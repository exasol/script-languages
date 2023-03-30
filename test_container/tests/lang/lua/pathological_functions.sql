CREATE lua SCALAR SCRIPT
sleep("sec" double)
RETURNS double AS

function sleep(s)
        start = os.time()
        while true do
                if start + s < os.time() then
                        break
                end
        end
end

function run(context)
        sleep(context.sec)
        return context.sec
end
/

CREATE lua SCALAR SCRIPT
mem_hog("mb" double)
RETURNS double AS

function run(ctx) 
	local a = string.rep('x', 1024*1024)
	local b = {}
	for i=1, ctx.mb do
			b[i] = a .. i
	end
	return #b
end
/

-- vim: ts=2:sts=2:sw=2
