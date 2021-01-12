using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD

include("param.jl")
include("genInitialWeights.jl")
include("genPlasticWeights.jl")
include("convertWgtIn2Out.jl")
include("genTarget.jl")
include("genStim.jl")
include("runinitial.jl")
include("runtrain.jl")
include("runtest.jl")
include("rls.jl")
include("funSample.jl")

#----------- initialization --------------#
w0Index, w0Weights, nc0 = genInitialWeights(p)
uavg, ns0, ustd = runinitial(p,w0Index, w0Weights, nc0,[])
wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut = genPlasticWeights(p, w0Index, nc0, ns0)
xtarg = genTarget(p,uavg,"zero")
stim = genStim(p)

#----------- save initialization --------------#
save("data/p.jld", "p", p)
save("data/w0Index.jld", "w0Index", w0Index)
save("data/w0Weights.jld", "w0Weights", w0Weights)
save("data/nc0.jld", "nc0", nc0)
save("data/stim.jld", "stim", stim)
save("data/xtarg.jld", "xtarg", xtarg)
save("data/wpIndexIn.jld", "wpIndexIn", wpIndexIn)
save("data/wpIndexOut.jld", "wpIndexOut", wpIndexOut)
save("data/wpIndexConvert.jld", "wpIndexConvert", wpIndexConvert)
save("data/wpWeightIn.jld", "wpWeightIn", wpWeightIn)
save("data/wpWeightOut.jld", "wpWeightOut", wpWeightOut)
save("data/ncpIn.jld", "ncpIn", ncpIn)
save("data/ncpOut.jld", "ncpOut", ncpOut)
