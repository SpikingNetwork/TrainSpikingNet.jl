using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD

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
ncpIn = load(joinpath(data_dir,"ncpIn.jld"))["ncpIn"]
ncpOut = load(joinpath(data_dir,"ncpOut.jld"))["ncpOut"]

wpWeightIn = transpose(dropdims(wpWeightIn, dims=2))

isnothing(p.seed) || Random.seed!(p.seed)

# --- load code --- #

include(joinpath(@__DIR__,"convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"runtrain.jl"))
include(joinpath(@__DIR__,"runtest.jl"))
include(joinpath(@__DIR__,"rls.jl"))

# --- set up correlation matrix --- #
P = Vector{Array{Float64,2}}(); 
Px = Vector{Array{Int64,1}}();

ci_numExcSyn = p.Lexc;
ci_numInhSyn = p.Linh;
ci_numSyn = ci_numExcSyn + ci_numInhSyn

# L2-penalty
Pinv_L2 = p.penlambda*one(zeros(ci_numSyn,ci_numSyn))
# row sum penalty
vec10 = [ones(ci_numExcSyn); zeros(ci_numInhSyn)];
vec01 = [zeros(ci_numExcSyn); ones(ci_numInhSyn)];
Pinv_rowsum = p.penmu*(vec10*vec10' + vec01*vec01')
# sum of penalties
Pinv = Pinv_L2 + Pinv_rowsum;
Pinv_norm = Pinv \ one(zeros(ci_numSyn,ci_numSyn))

for ci=1:Int(p.Ncells)
    # neurons presynaptic to ci
    push!(Px, wpIndexIn[ci,:]) 

    push!(P, copy(Pinv_norm));
end

# set up variables
invtauedecay = p.FloatPrecision(1/p.tauedecay)
invtauidecay = p.FloatPrecision(1/p.tauidecay)
invtaudecay_plastic = p.FloatPrecision(1/p.taudecay_plastic)

mu = zeros(p.FloatPrecision, p.Ncells)
mu[1:p.Ne] = (p.muemax-p.muemin)*rand(p.Ne) .+ p.muemin
mu[(p.Ne+1):p.Ncells] = (p.muimax-p.muimin)*rand(p.Ni) .+ p.muimin

thresh = zeros(p.FloatPrecision, p.Ncells)
thresh[1:p.Ne] .= p.threshe
thresh[(1+p.Ne):p.Ncells] .= p.threshi

invtau = zeros(p.FloatPrecision, p.Ncells)
invtau[1:p.Ne] .= 1/p.taue
invtau[(1+p.Ne):p.Ncells] .= 1/p.taui

maxTimes = round(Int,p.maxrate*p.train_time/1000)
times = zeros(p.Ncells,maxTimes)
ns = zeros(p.IntPrecision, p.Ncells)

forwardInputsE = zeros(p.FloatPrecision, p.Ncells)     # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = zeros(p.FloatPrecision, p.Ncells)     # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = zeros(p.FloatPrecision, p.Ncells)     # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = zeros(p.FloatPrecision, p.Ncells) # copy of forwardInputsE from previous time step
forwardInputsIPrev = zeros(p.FloatPrecision, p.Ncells) # copy of forwardInputsI from previous time step
forwardInputsPPrev = zeros(p.FloatPrecision, p.Ncells) # copy of forwardInputsP from previous time step
forwardSpike = zeros(p.FloatPrecision, p.Ncells)       # spikes emitted by each neuron at one time step
forwardSpikePrev = zeros(p.FloatPrecision, p.Ncells)   # copy of forwardSpike from previous time step

xedecay = zeros(p.FloatPrecision, p.Ncells)          # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = zeros(p.FloatPrecision, p.Ncells)          # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = zeros(p.FloatPrecision, p.Ncells)          # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = zeros(p.FloatPrecision, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = zeros(p.FloatPrecision, p.Ncells)                # synapse-filtered spikes (i.e. filtered version of forwardSpike)

bias = zeros(p.FloatPrecision, p.Ncells)             # total external input to neurons
lastSpike = Array{p.FloatPrecision}(undef, p.Ncells)  # last time a neuron spiked

plusone = p.FloatPrecision(1.0)
minusone = p.FloatPrecision(-1.0)

refrac = p.FloatPrecision(p.refrac)
vre = p.FloatPrecision(p.vre)

Px = Vector{Vector{p.IntPrecision}}(Px);
P = Vector{Matrix{p.FloatPrecision}}(P);
stim = Array{p.FloatPrecision}(stim);
xtarg = Array{p.FloatPrecision}(xtarg);
dt = p.FloatPrecision(p.dt)
ncpIn = Array{p.IntPrecision}(ncpIn)
w0Index = Array{p.IntPrecision}(w0Index)
w0Weights = Array{p.FloatPrecision}(w0Weights)
wpIndexIn = Array{p.IntPrecision}(wpIndexIn)
wpIndexConvert = Array{p.IntPrecision}(wpIndexConvert)
wpIndexOut = Array{p.IntPrecision}(wpIndexOut)
wpWeightIn = Array{p.FloatPrecision}(wpWeightIn);
wpWeightOut = Array{p.FloatPrecision}(wpWeightOut)

k = Vector{p.FloatPrecision}(undef, 2*p.L)
v = Vector{p.FloatPrecision}(undef, p.Ncells)

#----------- train the network --------------#

wpWeightIn, wpWeightOut = runtrain(p.nloop, p.learn_every, p.stim_on,
    p.stim_off, p.train_time, dt, p.Nsteps, p.Ncells, refrac, vre,
    invtauedecay, invtauidecay, invtaudecay_plastic, mu, thresh, invtau,
    ns, forwardInputsE, forwardInputsI, forwardInputsP, forwardInputsEPrev,
    forwardInputsIPrev, forwardInputsPPrev, forwardSpike, forwardSpikePrev,
    xedecay, xidecay, xpdecay, synInputBalanced, r, bias, lastSpike, plusone,
    minusone, k, v, P, Px, w0Index, w0Weights, nc0, stim, xtarg, wpIndexIn,
    wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut)

#----------- test the network --------------#
example_neurons = 25

vtotal_exc, vtotal_inh, vebal_exc, vibal_exc, vebal_inh, vibal_inh,
    vplastic_exc, vplastic_inh = runtest(p.stim_on, p.stim_off, dt,
    p.Nsteps, p.Ncells, refrac, vre, invtauedecay, invtauidecay,
    invtaudecay_plastic, mu, thresh, invtau, maxTimes, times, ns,
    forwardInputsE, forwardInputsI, forwardInputsP, forwardInputsEPrev,
    forwardInputsIPrev, forwardInputsPPrev, xedecay, xidecay, xpdecay,
    synInputBalanced, v, lastSpike, bias, example_neurons, w0Index, w0Weights,
    nc0, wpIndexOut, wpWeightOut, ncpOut, stim)

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

save(joinpath(data_dir,"wpWeightIn-trained.jld"), "wpWeightIn", collect(wpWeightIn))
save(joinpath(data_dir,"wpWeightOut-trained.jld"), "wpWeightOut", wpWeightOut)
