#using Pkg
#Pkg.add("SparseArrays")
#Pkg.instantiate()
#Pkg.resolve()
#Pkg.develop(path="../../")

using TrainSpikingNet
#using Revise

#if !isfile("my-data/param.jl")
#    mkdir("my-data")
#    cp(joinpath(dirname(pathof(TrainSpikingNet)), "param.jl"), "my-data/param.jl")
#end
#The parameters file is Julia code which sets various simulation variables using constants and user-defined plugins. To evaluate it, and save the pertinent data to a JLD2 file, use the param command:

p = param("Potjans_data");

p.dt  # the simulation time step in millisecods
#0.1

p.Ncells  # the number of neurons in the model
#4096

p.cellModel_file  # the plugin which defines membrane potential and spiking
#"/home/arthurb/.julia/packages/TrainSpikingNet/XYpdq/src/cellModel-LIF.jl"

#In addition to "param.jld2", alluded to above, there is also now a file called "rng-init.jld2" in your data folder. It contains the initial state of the random number generator used to initialize the model, which can be used to exactly reproduce an experiment.


#Now use config to load simulation code that is customized to your particular model architecture and machine hardware:
readdir("Potjans_data")
config("Potjans_data", :cpu) 



#conn_path = joinpath(dirname(pathof(TrainSpikingNet)), "contrib","genPotjansConnectivity.jl")
#@show(conn_path)
@show(p.genStaticWeights_file)
#p.genStaticWeights_file = conn_path
state = init();
