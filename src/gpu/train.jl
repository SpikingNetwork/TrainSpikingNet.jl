using LinearAlgebra, Random, JLD, Statistics, CUDA, NNlib, NNlibCUDA

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

# --- monitor resources used ---#
function monitor_resources(c::Channel)
  while true
    isopen(c) || break
    ipmitool = readlines(pipeline(`sudo ipmitool sensor`,
                                  `grep "PW Consumption"`,
                                  `cut -d'|' -f2`))
    top = readlines(pipeline(`top -b -n 2 -p $(getpid())`,
                             `tail -1`,
                             `awk '{print $9; print $10}'`))
    nvidiasmi = readlines(pipeline(`nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory --format=csv,noheader,nounits`))
    data = dropdims(sum(hcat([parse.(Float64, split(strip(x), ',')) for x in nvidiasmi]...),
                        dims=2), dims=2)
    println("total power used: ", strip(ipmitool[1]), " Watts\n",
            "CPU cores used by this process: ", strip(top[1]), "%\n",
            "CPU memory used by this process: ", strip(top[2]), "%\n",
            "GPU power used: ", data[1], " Watts\n",
            "GPU cores used: ", data[2], "%\n",
            "GPU memory used: ", data[3], "%")
    sleep(p.monitor_resources_used)
  end
end

if p.monitor_resources_used>0
  chnl = Channel(monitor_resources)
  sleep(60)
end

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
        plusone, minusone, k, den, e, delta, v, P, Px, w0Index, w0Weights, nc0,
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
            nothing, nothing, nothing, nothing, nothing, v, nothing, nothing, w0Index,
            w0Weights, nc0, stim, nothing, nothing, wpIndexOut, nothing,
            nothing, wpWeightOut, ncpOut, nothing, nothing)

        pcor = zeros(p.Ncells)
        for (index, ci) in enumerate(1:p.Ncells)
            xtarg_slice = convert(Array{Float64}, xtarg[:,ci])
            xtotal_slice = Array(xtotal[:,ci])
            pcor[index] = cor(xtarg_slice,xtotal_slice)
        end

        println("cor = ",mean(pcor))
    end

end

save(joinpath(data_dir,"wpWeightIn-trained.jld"), "wpWeightIn", Array(wpWeightIn))
save(joinpath(data_dir,"wpWeightOut-trained.jld"), "wpWeightOut", Array(wpWeightOut))

if p.monitor_resources_used>0
  sleep(60)
  close(chnl)
end
