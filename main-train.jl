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

#----------- load initialization --------------#
p = load("data/p.jld")["p"]
w0Index = load("data/w0Index.jld")["w0Index"]
w0Weights = load("data/w0Weights.jld")["w0Weights"]
nc0 = load("data/nc0.jld")["nc0"]
stim = load("data/stim.jld")["stim"]
xtarg = load("data/xtarg.jld")["xtarg"]
wpIndexIn = load("data/wpIndexIn.jld")["wpIndexIn"]
wpIndexOut = load("data/wpIndexOut.jld")["wpIndexOut"]
wpIndexConvert = load("data/wpIndexConvert.jld")["wpIndexConvert"]
wpWeightIn = load("data/wpWeightIn.jld")["wpWeightIn"]
wpWeightOut = load("data/wpWeightOut.jld")["wpWeightOut"]
ncpIn = load("data/ncpIn.jld")["ncpIn"]
ncpOut = load("data/ncpOut.jld")["ncpOut"]

#----------- train the network --------------#
wpWeightIn, wpWeightOut = runtrain(p,w0Index,w0Weights,nc0,stim,xtarg,wpIndexIn,wpIndexOut,wpIndexConvert,wpWeightIn,wpWeightOut,ncpIn,ncpOut)

#----------- test the network --------------#
times, ns, 
vtotal_exc, vtotal_inh, vebal_exc, vibal_exc, 
vebal_inh, vibal_inh, vplastic_exc, vplastic_inh = runtest(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim)

#----------- plot trained activities --------------#
timev = p.dt * collect(1:p.Nsteps)
timev_slice = collect(p.stim_off + p.learn_every: p.learn_every : p.train_time)
figure(figsize=(12,12))
for ii = 1:9
    subplot(3,3,ii)
    plot(timev, vtotal_exc[:,ii] .+ p.muemax, linewidth=0.5)
    plot(timev_slice, xtarg[:,ii] .+ p.muemax, linewidth=2)
    ylim([-2,2])
end
tight_layout()