using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD

data_dir = length(ARGS)>0 ? ARGS[1] : "."

include("struct.jl")
include(joinpath(data_dir,"param.jl"))
include(joinpath(@__DIR__,"genInitialWeights.jl"))
include(joinpath(@__DIR__,"genPlasticWeights.jl"))
include(joinpath(@__DIR__,"gpu","convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"genTarget.jl"))
include(joinpath(@__DIR__,"genStim.jl"))
include(joinpath(@__DIR__,"runinitial.jl"))
include(joinpath(@__DIR__,"funSample.jl"))

# set up variables
invtauedecay = 1/p.tauedecay
invtauidecay = 1/p.tauidecay

mu = zeros(p.Ncells)
mu[1:p.Ne] = (p.muemax-p.muemin)*rand(p.Ne) .+ p.muemin
mu[(p.Ne+1):p.Ncells] = (p.muimax-p.muimin)*rand(p.Ni) .+ p.muimin

thresh = zeros(p.Ncells)
thresh[1:p.Ne] .= p.threshe
thresh[(1+p.Ne):p.Ncells] .= p.threshi

invtau = zeros(p.Ncells)
invtau[1:p.Ne] .= 1/p.taue
invtau[(1+p.Ne):p.Ncells] .= 1/p.taui

maxTimes = round(Int,p.maxrate*p.train_time/1000)
times = zeros(p.Ncells,maxTimes)
ns = zeros(Int,p.Ncells)

forwardInputsE = zeros(p.Ncells) #summed weight of incoming E spikes
forwardInputsI = zeros(p.Ncells)
forwardInputsEPrev = zeros(p.Ncells) #as above, for previous timestep
forwardInputsIPrev = zeros(p.Ncells)

xedecay = zeros(p.Ncells)
xidecay = zeros(p.Ncells)

v = p.threshe*rand(p.Ncells) #membrane voltage 

lastSpike = -100.0*ones(p.Ncells) #time of last spike
  
uavg = zeros(p.Ncells)
utmp = zeros(p.Nsteps - Int(1000/p.dt),1000)
bias = zeros(p.Ncells)

#----------- initialization --------------#
w0Index, w0Weights, nc0 = genInitialWeights(p)

uavg, ns0, ustd = runinitial(train_time, dt, Nsteps, Ncells, Ne, refrac, vre,
    invtauedecay, invtauidecay, mu, thresh, invtau, maxTimes, times, ns, forwardInputsE,
    forwardInputsI, forwardInputsEPrev, forwardInputsIPrev, xedecay, xidecay,
    v, lastSpike, uavg, utmp, bias, w0Index, w0Weights, nc0)

wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut =
    genPlasticWeights(p, w0Index, nc0, ns0)

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
