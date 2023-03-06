using TrainSpikingNet, Test, CUDA

testgpu = true
try
    n = CUDA.ndevices()
    @info string("found ", n, " GPUs")
    global testgpu = true
catch
    global testgpu = false
    @warn "no GPU found"
end

#The parameters file is Julia code which sets various simulation variables using constants and user-defined plugins. To evaluate it, and save the pertinent data to a JLD2 file, use the param command:
p = param("PotjansParam");
readdir("PotjansParam")
config("PotjansParam", :cpu) 
state = init()
@test length(state)>0
weights = train(nloops=100);
@test length(weights)>0
readdir("PotjansParam")
activities = test(ntrials=100);
@test length(activities.nss[1])>0
readdir("PotjansParam")
#plot("PotjansParam/test.jld2", ineurons_to_plot=[1,5,9,13])


