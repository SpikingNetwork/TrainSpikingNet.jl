using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD
using CUDA, NNlib, NNlibCUDA

data_dir = length(ARGS)>0 ? ARGS[1] : "."

#----------- load initialization --------------#
include(joinpath(dirname(@__DIR__),"struct.jl"))
p = load(joinpath(data_dir,"p.jld"))["p"]
w0Index = load(joinpath(data_dir,"w0Index.jld"))["w0Index"]
w0Weights = load(joinpath(data_dir,"w0Weights.jld"))["w0Weights"]
nc0 = load(joinpath(data_dir,"nc0.jld"))["nc0"]
stim = load(joinpath(data_dir,"stim.jld"))["stim"]
xtarg = load(joinpath(data_dir,"xtarg.jld"))["xtarg"]
wpIndexIn = load(joinpath(data_dir,"wpIndexIn.jld"))["wpIndexIn"]
wpIndexOut = load(joinpath(data_dir,"wpIndexOut.jld"))["wpIndexOut"]
wpIndexConvert = load(joinpath(data_dir,"wpIndexConvert.jld"))["wpIndexConvert"]
wpWeightIn = load(joinpath(data_dir,"wpWeightIn.jld"))["wpWeightIn"]
wpWeightOut = load(joinpath(data_dir,"wpWeightOut.jld"))["wpWeightOut"]
ncpOut = load(joinpath(data_dir,"ncpOut.jld"))["ncpOut"]

isnothing(p.seed) || Random.seed!(p.seed)

# --- load code --- #

include(joinpath(@__DIR__,"convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"runtrain.jl"))
include(joinpath(@__DIR__,"runtest.jl"))
include(joinpath(@__DIR__,"rls.jl"))

#--- set up correlation matrix ---#
ci_numExcSyn = p.Lexc;
ci_numInhSyn = p.Linh;
ci_numSyn = ci_numExcSyn + ci_numInhSyn

# neurons presynaptic to ci
Px = wpIndexIn'

# L2-penalty
Pinv_L2 = p.penlambda*one(zeros(ci_numSyn,ci_numSyn))
# row sum penalty
vec10 = [ones(ci_numExcSyn); zeros(ci_numInhSyn)];
vec01 = [zeros(ci_numExcSyn); ones(ci_numInhSyn)];
Pinv_rowsum = p.penmu*(vec10*vec10' + vec01*vec01')
# sum of penalties
Pinv = Pinv_L2 + Pinv_rowsum;
P = Array{Float64}(undef, (p.Lexc+p.Linh, p.Lexc+p.Linh, p.Ncells)); 
P .= Pinv \ one(zeros(ci_numSyn,ci_numSyn));

# set up variables
mu = CUDA.zeros(p.FloatPrecision, p.Ncells)
mu[1:p.Ne] = (p.muemax-p.muemin)*rand(p.Ne) .+ p.muemin
mu[(p.Ne+1):p.Ncells] = (p.muimax-p.muimin)*rand(p.Ni) .+ p.muimin

thresh = CUDA.zeros(p.FloatPrecision, p.Ncells)
thresh[1:p.Ne] .= p.threshe
thresh[(1+p.Ne):p.Ncells] .= p.threshi

invtau = CUDA.zeros(p.FloatPrecision, p.Ncells)
invtau[1:p.Ne] .= 1/p.taue
invtau[(1+p.Ne):p.Ncells] .= 1/p.taui

maxTimes = round(Int,p.maxrate*p.train_time/1000)
times = zeros(p.Ncells,maxTimes)   ### never used?
ns = CUDA.zeros(p.IntPrecision, p.Ncells)

forwardInputsE = CUDA.zeros(p.FloatPrecision, p.Ncells+1)  # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = CUDA.zeros(p.FloatPrecision, p.Ncells+1)  # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = CUDA.zeros(p.FloatPrecision, p.Ncells+1)  # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = CUDA.zeros(p.FloatPrecision, p.Ncells+1) # copy of forwardInputsE from previous time step
forwardInputsIPrev = CUDA.zeros(p.FloatPrecision, p.Ncells+1) # copy of forwardInputsI from previous time step
forwardInputsPPrev = CUDA.zeros(p.FloatPrecision, p.Ncells+1) # copy of forwardInputsP from previous time step
forwardSpike = CUDA.zeros(p.FloatPrecision, p.Ncells) # spikes emitted by each neuron at one time step
forwardSpikePrev = CUDA.zeros(p.FloatPrecision, p.Ncells) # copy of forwardSpike from previous time step

xedecay = CUDA.zeros(p.FloatPrecision, p.Ncells)  # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = CUDA.zeros(p.FloatPrecision, p.Ncells)  # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = CUDA.zeros(p.FloatPrecision, p.Ncells)  # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = CUDA.zeros(p.FloatPrecision, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = CUDA.zeros(p.FloatPrecision, p.Ncells)      # synapse-filtered spikes (i.e. filtered version of forwardSpike)

bias = CUDA.zeros(p.FloatPrecision, p.Ncells)   # total external input to neurons
lastSpike = CuArray{p.FloatPrecision}(undef, p.Ncells)  # last time a neuron spiked

bnotrefrac = CuVector{Bool}(undef, p.Ncells)
bspike = CuVector{Bool}(undef, p.Ncells)
plusone = p.FloatPrecision(1.0)
minusone = p.FloatPrecision(-1.0)

refrac = p.FloatPrecision(p.refrac)
vre = p.FloatPrecision(p.vre)

Px = CuArray{p.IntPrecision}(Px);
P = CuArray{p.FloatPrecision}(P);
stim = CuArray{p.FloatPrecision}(stim);
xtarg = CuArray{p.FloatPrecision}(xtarg);
invtauedecay = p.FloatPrecision(1/p.tauedecay)
invtauidecay = p.FloatPrecision(1/p.tauidecay)
invtaudecay_plastic = p.FloatPrecision(1/p.taudecay_plastic)
dt = p.FloatPrecision(p.dt)
w0Index = CuArray{p.IntPrecision}(w0Index)
w0Weights = CuArray{p.FloatPrecision}(w0Weights)
wpIndexIn = CuArray{p.IntPrecision}(wpIndexIn)
wpIndexConvert = CuArray{p.IntPrecision}(wpIndexConvert)
wpIndexOut = CuArray{p.IntPrecision}(wpIndexOut)
wpWeightIn = CuArray{p.FloatPrecision}(wpWeightIn);
wpWeightOut = CuArray{p.FloatPrecision}(wpWeightOut)
times = CuArray{p.FloatPrecision}(times)

k = CuArray{p.FloatPrecision}(undef, 2*p.L, 1, p.Ncells)
den = CuArray{p.FloatPrecision}(undef, 1, 1, p.Ncells)
e = CuArray{p.FloatPrecision}(undef, 1, 1, p.Ncells)
v = CuVector{p.FloatPrecision}(undef, p.Ncells)

#----------- train the network --------------#

wpWeightIn, wpWeightOut = runtrain(p.nloop, p.learn_every, p.stim_on,
    p.stim_off, p.train_time, p.dt, p.Nsteps, p.Ncells, p.L, refrac, vre,
    invtauedecay, invtauidecay, invtaudecay_plastic, mu, thresh, invtau,
    ns, forwardInputsE, forwardInputsI, forwardInputsP, forwardInputsEPrev,
    forwardInputsIPrev, forwardInputsPPrev, forwardSpike, forwardSpikePrev,
    xedecay, xidecay, xpdecay, synInputBalanced, r, bias, lastSpike,
    bnotrefrac, bspike, plusone, minusone, k, den, e, v, P, Px, w0Index,
    w0Weights, nc0, stim, xtarg, wpIndexIn, wpIndexOut, wpIndexConvert,
    wpWeightIn, wpWeightOut, ncpOut, cukernelEI, cukernelP)

#----------- test the network --------------#
example_neurons = 25

vtotal_exc, vtotal_inh, vebal_exc, vibal_exc, vebal_inh, vibal_inh,
    vplastic_exc, vplastic_inh = runtest(p.stim_on, p.stim_off, dt,
    p.Nsteps, p.Ncells, refrac, vre, invtauedecay, invtauidecay,
    invtaudecay_plastic, mu, thresh, invtau, maxTimes,
    times, ns, forwardInputsE, forwardInputsI, forwardInputsP,
    forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
    xedecay, xidecay, xpdecay, synInputBalanced, v, lastSpike, bias,
    example_neurons, w0Index, w0Weights, nc0, wpIndexOut, wpWeightOut,
    ncpOut, stim)

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

save(joinpath(data_dir,"wpWeightIn-trained.jld"), "wpWeightIn", Array(wpWeightIn))
save(joinpath(data_dir,"wpWeightOut-trained.jld"), "wpWeightOut", Array(wpWeightOut))
