using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD

data_dir = length(ARGS)>0 ? ARGS[1] : "."

BLAS.set_num_threads(8)

# --- load code --- #
kind=:test
include(joinpath(@__DIR__,"cpu","loop.jl"))
include(joinpath(@__DIR__,"cpu","funRollingAvg.jl"))

#----------- load initialization --------------#
include(joinpath(@__DIR__,"struct.jl"))
p = load(joinpath(data_dir,"p.jld"))["p"]
w0Index = load(joinpath(data_dir,"w0Index.jld"))["w0Index"]
w0Weights = load(joinpath(data_dir,"w0Weights.jld"))["w0Weights"]
nc0 = load(joinpath(data_dir,"nc0.jld"))["nc0"]
stim = load(joinpath(data_dir,"stim.jld"))["stim"]
xtarg = load(joinpath(data_dir,"xtarg.jld"))["xtarg"]
wpIndexOut = load(joinpath(data_dir,"wpIndexOut.jld"))["wpIndexOut"]
wpWeightOut = load(joinpath(data_dir,"wpWeightOut-trained.jld"))["wpWeightOut"]
ncpIn = load(joinpath(data_dir,"ncpIn.jld"))["ncpIn"]
ncpOut = load(joinpath(data_dir,"ncpOut.jld"))["ncpOut"]

isnothing(p.seed) || Random.seed!(p.seed)

# --- set up variables --- #
include(joinpath(@__DIR__,"cpu","variables.jl"))

#----------- test the network --------------#
example_neurons = 25

vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell,
    vibal_inhcell, vplastic_exccell, vplastic_inhcell = loop_test(p.learn_every,
    p.stim_on, p.stim_off, p.train_time, dt, p.Nsteps, p.Ncells, nothing,
    refrac, vre, invtauedecay, invtauidecay, invtaudecay_plastic, mu,
    thresh, invtau, ns, forwardInputsE, forwardInputsI, forwardInputsP,
    forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev, nothing,
    nothing, xedecay, xidecay, xpdecay, synInputBalanced, r, bias,
    example_neurons, lastSpike, nothing, nothing, nothing, v, nothing,
    nothing, w0Index, w0Weights, nc0, stim, nothing, nothing, wpIndexOut,
    nothing, nothing, wpWeightOut, nothing, ncpOut, nothing, nothing)

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
