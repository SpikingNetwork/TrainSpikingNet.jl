using TrainSpikingNet, Test, JLD2, SymmetricFormats, CUDA

testgpu = true
try
    n = CUDA.ndevices()
    @info string("found ", n, " GPUs")
    global testgpu = true
catch
    global testgpu = false
    @warn "no GPU found"
end

#include("Potjans_data/param.jl")
#The parameters file is Julia code which sets various simulation variables using constants and user-defined plugins. To evaluate it, and save the pertinent data to a JLD2 file, use the param command:
p = param("Potjans_data");
readdir("Potjans_data")
config("Potjans_data", :gpu) 
state = init()
weights = train(nloops=100);
readdir("Potjans_data")
activities = test(ntrials=100);
activities.nss[1]
readdir("Potjans_data")
plot("Potjans_data/test.jld2", ineurons_to_plot=[1,5,9,13])


