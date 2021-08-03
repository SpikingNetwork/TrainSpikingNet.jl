using Distributions
using LinearAlgebra
using Random
using JLD
using CUDA, NNlib, NNlibCUDA

data_dir = length(ARGS)>0 ? ARGS[1] : "."

CUDA.allowscalar(false)

#----------- load initialization --------------#
include(joinpath(dirname(@__DIR__),"struct.jl"))
include(joinpath(data_dir,"param.jl"))
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
kind=:train
include(joinpath(@__DIR__,"convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"loop.jl"))
include(joinpath(@__DIR__,"rls.jl"))
if p.performance_interval>0
    kind=:test
    include(joinpath(@__DIR__,"loop.jl"))
    include(joinpath(@__DIR__,"funRollingAvg.jl"))
end

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

# --- set up variables --- #
include(joinpath(@__DIR__,"variables.jl"))
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

#----------- train the network --------------#
for iloop =1:p.nloop
    println("Loop no. ",iloop) 

    start_time = time()

    loop_train(p.learn_every, p.stim_on, p.stim_off, p.train_time, dt,
        p.Nsteps, p.Ncells, p.L, nothing, refrac, vre, invtauedecay,
        invtauidecay, invtaudecay_plastic, mu, thresh, invtau, ns,
        forwardInputsE, forwardInputsI, forwardInputsP, forwardInputsEPrev,
        forwardInputsIPrev, forwardInputsPPrev, forwardSpike,
        forwardSpikePrev, xedecay, xidecay, xpdecay, synInputBalanced,
        synInput, r, bias, nothing, nothing, lastSpike, bnotrefrac, bspike,
        plusone, minusone, k, den, e, v, P, Px, w0Index, w0Weights, nc0,
        stim, xtarg, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn,
        wpWeightOut, ncpOut, nothing, nothing)

    elapsed_time = time()-start_time
    println("elapsed time: ",elapsed_time, " sec")
    println("firing rate: ",mean(ns)/(dt/1000*p.Nsteps), " Hz")

    # test performance
    if (p.performance_interval>0) && mod(iloop,p.performance_interval) == 0

        xtotal, _ = loop_test(
            p.learn_every, p.stim_on, p.stim_off, p.train_time, dt, p.Nsteps,
            p.Ncells, p.L, nothing, refrac, vre, invtauedecay, invtauidecay,
            invtaudecay_plastic, mu, thresh, invtau, ns, forwardInputsE,
            forwardInputsI, forwardInputsP, forwardInputsEPrev,
            forwardInputsIPrev, forwardInputsPPrev, nothing, nothing,
            xedecay, xidecay, xpdecay, synInputBalanced, synInput, r, bias,
            p.wid, p.example_neurons, lastSpike, bnotrefrac, bspike, nothing,
            nothing, nothing, nothing, nothing, v, nothing, nothing, w0Index,
            w0Weights, nc0, stim, nothing, nothing, wpIndexOut, nothing,
            nothing, wpWeightOut, ncpOut, nothing, nothing)

        pcor = zeros(p.Ncells)
        for (index, ci) in enumerate(1:p.Ncells)
            xtarg_slice = Array(xtarg[:,ci])
            xtotal_slice = Array(xtotal[:,ci])
            pcor[index] = cor(xtarg_slice,xtotal_slice)
        end

        println("cor = ",mean(pcor))
    end

end

save(joinpath(data_dir,"wpWeightIn-trained.jld"), "wpWeightIn", Array(wpWeightIn))
save(joinpath(data_dir,"wpWeightOut-trained.jld"), "wpWeightOut", Array(wpWeightOut))
