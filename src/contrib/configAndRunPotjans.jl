using TrainSpikingNet
using Revise
if !isfile("Potjans_data2/param.jl")
    mkdir("Potjans_data2")
    cp(joinpath(dirname(pathof(TrainSpikingNet)), "param.jl"), "Potjans_data/param.jl")
end
#The parameters file is Julia code which sets various simulation variables using constants and user-defined plugins. To evaluate it, and save the pertinent data to a JLD2 file, use the param command:
p = param("Potjans_data2");
readdir("Potjans_data2")
config("Potjans_data2", :cpu) 
state = init()
weights = train(nloops=100);
readdir("Potjans_data2")
activities = test(ntrials=100);
activities.nss[1]
readdir("Potjans_data2")
plot("Potjans_data2/test.jld2", ineurons_to_plot=[1,5,9,13])


