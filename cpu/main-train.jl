using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD

data_dir = length(ARGS)>0 ? ARGS[1] : "."

BLAS.set_num_threads(8)

Random.seed!(1)

include(joinpath(data_dir,"param.jl"))
include("genInitialWeights.jl")
include("genPlasticWeights.jl")
include("convertWgtIn2Out.jl")
include("genTarget.jl")
include("genStim.jl")
#include("runinitial.jl")
include("runtrain.jl")
include("runtest.jl")
include("rls.jl")
include("funSample.jl")

#----------- load initialization --------------#
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

# --- set up correlation matrix --- #
P = Vector{Array{Float64,2}}(); 
Px = Vector{Array{Int64,1}}();

ci_numExcSyn = p.Lexc;
ci_numInhSyn = p.Linh;
ci_numSyn = ci_numExcSyn + ci_numInhSyn

# L2-penalty
Pinv_L2 = penlambda*one(zeros(ci_numSyn,ci_numSyn))
# row sum penalty
vec10 = [ones(ci_numExcSyn); zeros(ci_numInhSyn)];
vec01 = [zeros(ci_numExcSyn); ones(ci_numInhSyn)];
Pinv_rowsum = penmu*(vec10*vec10' + vec01*vec01')
# sum of penalties
Pinv = Pinv_L2 + Pinv_rowsum;
Pinv_norm = Pinv \ one(zeros(ci_numSyn,ci_numSyn))

for ci=1:Int(Ncells)
    # neurons presynaptic to ci
    push!(Px, wpIndexIn[ci,:]) 

    push!(P, copy(Pinv_norm));
end

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
