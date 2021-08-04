using LinearAlgebra, Random, JLD, Statistics

data_dir = length(ARGS)>0 ? ARGS[1] : "."

BLAS.set_num_threads(1)

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
ncpIn = load(joinpath(data_dir,"ncpIn.jld"))["ncpIn"]
ncpOut = load(joinpath(data_dir,"ncpOut.jld"))["ncpOut"]

wpWeightIn = transpose(dropdims(wpWeightIn, dims=2))

isnothing(p.seed) || Random.seed!(p.seed)

# --- load code --- #
kind=:train
include(joinpath(@__DIR__,"convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"loop.jl"))
include(joinpath(@__DIR__,"rls.jl"))
if p.performance_interval>0
    kind=:test
    include(joinpath(@__DIR__,"loop.jl"))
end

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

# --- set up variables --- #
include(joinpath(@__DIR__,"variables.jl"))
Px = Vector{Vector{p.IntPrecision}}(Px);
P = Vector{Matrix{p.FloatPrecision}}(P);
stim = Array{p.FloatPrecision}(stim);
xtarg = Array{p.FloatPrecision}(xtarg);
ncpIn = Array{p.IntPrecision}(ncpIn)
w0Index = Array{p.IntPrecision}(w0Index)
w0Weights = Array{p.FloatPrecision}(w0Weights)
wpIndexIn = Array{p.IntPrecision}(wpIndexIn)
wpIndexConvert = Array{p.IntPrecision}(wpIndexConvert)
wpIndexOut = Array{p.IntPrecision}(wpIndexOut)
wpWeightIn = Array{p.FloatPrecision}(wpWeightIn);
wpWeightOut = Array{p.FloatPrecision}(wpWeightOut)

#----------- train the network --------------#
for iloop =1:p.nloop
    println("Loop no. ",iloop) 

    start_time = time()

    loop_train(p.learn_every, p.stim_on, p.stim_off, p.train_time, dt,
        p.Nsteps, p.Ncells, nothing, refrac, vre, invtauedecay,
        invtauidecay, invtaudecay_plastic, mu, thresh, invtau, nothing,
        nothing, ns, forwardInputsE, forwardInputsI, forwardInputsP,
        forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
        forwardSpike, forwardSpikePrev, xedecay, xidecay, xpdecay,
        synInputBalanced, synInput, r, bias, nothing, nothing, lastSpike,
        plusone, k, v, P, Px, w0Index, w0Weights, nc0, stim, xtarg, wpIndexIn,
        wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut,
        nothing, nothing)

    elapsed_time = time()-start_time
    println("elapsed time: ",elapsed_time, " sec")
    println("firing rate: ",mean(ns)/(dt/1000*p.Nsteps), " Hz")

    # test performance
    if (p.performance_interval>0) && mod(iloop,p.performance_interval) == 0

        xtotal, _ = loop_test(
            p.learn_every, p.stim_on, p.stim_off, p.train_time, dt,
            p.Nsteps, p.Ncells, nothing, refrac, vre, invtauedecay,
            invtauidecay, invtaudecay_plastic, mu, thresh, invtau, maxTimes,
            times, ns, forwardInputsE, forwardInputsI, forwardInputsP,
            forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
            nothing, nothing, xedecay, xidecay, xpdecay, synInputBalanced,
            synInput, nothing, bias, p.wid, p.example_neurons, lastSpike,
            nothing, nothing, v, nothing, nothing, w0Index, w0Weights,
            nc0, stim, nothing, nothing, wpIndexOut, nothing, nothing,
            wpWeightOut, nothing, ncpOut, nothing, nothing)

        pcor = zeros(p.Ncells)
        for (index, ci) in enumerate(1:p.Ncells)
            xtarg_slice = @view xtarg[:,ci]
            xtotal_slice = @view xtotal[:,ci]
            pcor[index] = cor(xtarg_slice,xtotal_slice)
        end

        println("cor = ",mean(pcor))
    end

end # end loop over trainings

save(joinpath(data_dir,"wpWeightIn-trained.jld"), "wpWeightIn", collect(wpWeightIn))
save(joinpath(data_dir,"wpWeightOut-trained.jld"), "wpWeightOut", wpWeightOut)
