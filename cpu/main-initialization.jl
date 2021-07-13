using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD

data_dir = length(ARGS)>0 ? ARGS[1] : "."

Random.seed!(1)

include(joinpath(data_dir,"param.jl"))
include("genInitialWeights.jl")
include("genPlasticWeights.jl")
include("convertWgtIn2Out.jl")
include("genTarget.jl")
include("genStim.jl")
include("runinitial.jl")
#include("runtrain.jl")
#include("runtest.jl")
#include("rls.jl")
include("funSample.jl")

#----------- initialization --------------#
w0Index, w0Weights, nc0 = genInitialWeights(p)
uavg, ns0, ustd = runinitial(p,w0Index, w0Weights, nc0,[])
wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut = genPlasticWeights(p, w0Index, nc0, ns0)
xtarg = genTarget(p,uavg,"zero")
stim = genStim(p)

#----------- save initialization --------------#
save(joinpath(data_dir,"p.jld"), "p", p)
save(joinpath(data_dir,"w0Index.jld"), "w0Index", w0Index)
save(joinpath(data_dir,"w0Weights.jld"), "w0Weights", w0Weights)
save(joinpath(data_dir,"nc0.jld"), "nc0", nc0)
save(joinpath(data_dir,"stim.jld"), "stim", stim)
save(joinpath(data_dir,"xtarg.jld"), "xtarg", xtarg)
save(joinpath(data_dir,"wpIndexIn.jld"), "wpIndexIn", wpIndexIn)
save(joinpath(data_dir,"wpIndexOut.jld"), "wpIndexOut", wpIndexOut)
save(joinpath(data_dir,"wpIndexConvert.jld"), "wpIndexConvert", wpIndexConvert)
save(joinpath(data_dir,"wpWeightIn.jld"), "wpWeightIn", wpWeightIn)
save(joinpath(data_dir,"wpWeightOut.jld"), "wpWeightOut", wpWeightOut)
save(joinpath(data_dir,"ncpIn.jld"), "ncpIn", ncpIn)
save(joinpath(data_dir,"ncpOut.jld"), "ncpOut", ncpOut)
