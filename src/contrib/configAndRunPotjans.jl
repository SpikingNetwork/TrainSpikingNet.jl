using TrainSpikingNet

if !isfile("Potjans_data/param.jl")
    mkdir("Potjans_data")
    cp(joinpath(dirname(pathof(TrainSpikingNet)), "param.jl"), "Potjans_data/param.jl")
end
#The parameters file is Julia code which sets various simulation variables using constants and user-defined plugins. To evaluate it, and save the pertinent data to a JLD2 file, use the param command:
p = param("Potjans_data");
readdir("Potjans_data")
config("Potjans_data", :cpu) 
if false
    # this works but it is slow.
    # probably because of sparse arrays not being as good with GPU
    config("Potjans_data", :gpu) 
end



state = init()
function convert_ragged_arraytodense(times)
    converttimes = []
    convertnodes = []
    for (ind,i) in enumerate(times)
        for t in i
            append!(converttimes,i)
            append!(convertnodes,ind)
        end
    end
    (converttimes,convertnodes)
end


path = joinpath(dirname(pathof(TrainSpikingNet)), "cpu/variables.jl")
include(path)    
w0Index, w0Weights, nc0 = TrainSpikingNet.genStaticWeights(p.genStaticWeights_args)
rateX = TrainSpikingNet.genRateX(p.genRateX_args)
times = []
for i in 1:p.Ncells
    push!(times,Float32[])
end
itask = 1
uavg0, ns0, ustd0 = TrainSpikingNet.loop_init(itask,
nothing, nothing, p.stim_off, p.train_time, p.dt,
p.Nsteps, p.Ncells, p.Ne, nothing, p.LX, p.refrac,
learn_step, invtau_bale, invtau_bali, nothing, X_bal, nothing,
nothing, ns, nothing, nsX, inputsE, inputsI, nothing,
inputsEPrev, inputsIPrev, nothing, nothing, nothing, nothing, nothing,
u_bale, u_bali, nothing, u_bal, u, nothing, nothing,
X, nothing, nothing, lastSpike, nothing, nothing, nothing, nothing,
nothing, v, p.rng, noise, rndX, sig, nothing, w0Index,
w0Weights, nc0, nothing, nothing, nothing, nothing, nothing, nothing,
nothing, nothing, nothing, nothing, uavg, ustd, rateX, cellModel_args)

(times,nodes) = convert_ragged_arraytodense(times)
plot("my-data/test.jld2", ineurons_to_plot=[1,5,9,13])

#end

