using Distributions
using LinearAlgebra
using Random
using JLD

data_dir = length(ARGS)>0 ? ARGS[1] : "."

Random.seed!(1)

# --- load code --- #
kind=:init
include(joinpath(@__DIR__,"genInitialWeights.jl"))
include(joinpath(@__DIR__,"genPlasticWeights.jl"))
include(joinpath(@__DIR__,"gpu","convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"genTarget.jl"))
include(joinpath(@__DIR__,"genStim.jl"))
include(joinpath(@__DIR__,"cpu","loop.jl"))
include(joinpath(@__DIR__,"funSample.jl"))

# --- set up variables --- #
include(joinpath(@__DIR__,"struct.jl"))
include(joinpath(data_dir,"param.jl"))
include(joinpath(@__DIR__,"cpu","variables.jl"))

#----------- initialization --------------#
w0Index, w0Weights, nc0 = genInitialWeights(p)

uavg, ns0, ustd = loop_init(nothing, nothing, nothing, p.train_time, dt,
    p.Nsteps, p.Ncells, p.Ne, refrac, vre, invtauedecay, invtauidecay,
    nothing, mu, thresh, invtau, ns, forwardInputsE, forwardInputsI,
    nothing, forwardInputsEPrev, forwardInputsIPrev, nothing, nothing,
    nothing, xedecay, xidecay, nothing, nothing, nothing, bias, nothing,
    lastSpike, nothing, nothing, nothing, v, nothing, nothing, w0Index,
    w0Weights, nc0, nothing, nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, uavg, utmp)

wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut =
    genPlasticWeights(p, w0Index, nc0, ns0)

xtarg = genTarget(p,uavg,"zero")
stim = genStim(p)

#----------- save initialization --------------#
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
