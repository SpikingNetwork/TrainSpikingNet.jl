using LinearAlgebra, Random, JLD, Statistics, ArgParse

s = ArgParseSettings()

@add_arg_table! s begin
    "--nloops", "-n"
        help = "number of iterations to train"
        arg_type = Int
        default = 1
        metavar = "N"
    "--performance_interval", "-p"
        help = "measure correlation every P training loops.  default is never"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
        metavar = "P"
    "--save_checkpoints", "-c"
        help = "save learned weights every C training loops.  default is to only save the last loop"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
        metavar = "C"
    "--restore_from_checkpoint", "-r"
        help = "continue training from checkpoint R.  default is to start from the beginning"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
        metavar = "R"
    "--monitor_resources_used", "-m"
        help = "measure power, cores, and memory usage every R seconds.  default is never"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
        metavar = "M"
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
end

parsed_args = parse_args(s)

BLAS.set_num_threads(1)

#----------- load initialization --------------#
include(joinpath(dirname(@__DIR__),"struct.jl"))
p = load(joinpath(parsed_args["data_dir"],"p.jld"))["p"]
w0Index = load(joinpath(parsed_args["data_dir"],"w0Index.jld"))["w0Index"]
w0Weights = load(joinpath(parsed_args["data_dir"],"w0Weights.jld"))["w0Weights"]
nc0 = load(joinpath(parsed_args["data_dir"],"nc0.jld"))["nc0"]
stim = load(joinpath(parsed_args["data_dir"],"stim.jld"))["stim"]
xtarg = load(joinpath(parsed_args["data_dir"],"xtarg.jld"))["xtarg"]
wpIndexIn = load(joinpath(parsed_args["data_dir"],"wpIndexIn.jld"))["wpIndexIn"]
wpIndexOut = load(joinpath(parsed_args["data_dir"],"wpIndexOut.jld"))["wpIndexOut"]
wpIndexConvert = load(joinpath(parsed_args["data_dir"],"wpIndexConvert.jld"))["wpIndexConvert"]
if isnothing(parsed_args["restore_from_checkpoint"])
    R=0
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn.jld"))["wpWeightIn"]
    wpWeightOut = load(joinpath(parsed_args["data_dir"],"wpWeightOut.jld"))["wpWeightOut"]
    wpWeightIn = transpose(dropdims(wpWeightIn, dims=2))
else
    R = parsed_args["restore_from_checkpoint"]
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld"))["wpWeightIn"]
    wpWeightOut = load(joinpath(parsed_args["data_dir"],"wpWeightOut-ckpt$R.jld"))["wpWeightOut"]
end
ncpIn = load(joinpath(parsed_args["data_dir"],"ncpIn.jld"))["ncpIn"]
ncpOut = load(joinpath(parsed_args["data_dir"],"ncpOut.jld"))["ncpOut"]

isnothing(p.seed) || Random.seed!(p.rng, p.seed)

# --- load code --- #
macro maybethread(loop)
  if Threads.nthreads()>1
    quote Threads.@threads $(Expr(loop.head,
                             Expr(loop.args[1].head, esc.(loop.args[1].args)...),
                             esc(loop.args[2]))); end
  else
    @warn "running single threaded"
    quote $(esc(loop)); end 
  end
end

kind=:train
include(joinpath(@__DIR__,"convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"loop.jl"))
include(joinpath(@__DIR__,"rls.jl"))
if !isnothing(parsed_args["performance_interval"])
    kind=:train_test
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
Pinv_norm = Pinv \ I

for ci=1:p.Ncells
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
nc0 = Array{p.IntPrecision}(nc0)
ncpIn = Array{p.IntPrecision}(ncpIn)
ncpOut = Array{p.IntPrecision}(ncpOut)
w0Index = Array{p.IntPrecision}(w0Index)
w0Weights = Array{p.FloatPrecision}(w0Weights)
wpIndexIn = Array{p.IntPrecision}(wpIndexIn)
wpIndexConvert = Array{p.IntPrecision}(wpIndexConvert)
wpIndexOut = Array{p.IntPrecision}(wpIndexOut)
wpWeightIn = Array{p.FloatPrecision}(wpWeightIn);
wpWeightOut = Array{p.FloatPrecision}(wpWeightOut)

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
    println("total power used: ", strip(ipmitool[1]), " Watts\n",
            "CPU cores used by this process: ", strip(top[1]), "%\n",
            "CPU memory used by this process: ", strip(top[2]), '%')
    sleep(parsed_args["monitor_resources_used"])
  end
end

if !isnothing(parsed_args["monitor_resources_used"])
  chnl = Channel(monitor_resources)
  sleep(60)
end

#----------- train the network --------------#
for iloop = R.+(1:parsed_args["nloops"])
    println("Loop no. ",iloop) 

    start_time = time()

    if isnothing(parsed_args["performance_interval"]) ||
       mod(iloop,parsed_args["performance_interval"]) != 0

        loop_train(
            p.learn_every, p.stim_on, p.stim_off, p.train_time, dt,
            p.Nsteps, p.Ncells, nothing, refrac, vre, invtauedecay,
            invtauidecay, invtaudecay_plastic, mu, thresh, invtau, nothing,
            nothing, ns, forwardInputsE, forwardInputsI, forwardInputsP,
            forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
            forwardSpike, forwardSpikePrev, xedecay, xidecay, xpdecay,
            synInputBalanced, synInput, r, bias, nothing, nothing, lastSpike,
            plusone, k, v, p.rng, noise, sig, P, Px, w0Index, w0Weights, nc0, stim,
            xtarg, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut,
            ncpIn, ncpOut, nothing, nothing)
    else
        _, _, xtotal, _ = loop_train_test(
            p.learn_every, p.stim_on, p.stim_off, p.train_time, dt,
            p.Nsteps, p.Ncells, nothing, refrac, vre, invtauedecay,
            invtauidecay, invtaudecay_plastic, mu, thresh, invtau, maxTimes,
            times, ns, forwardInputsE, forwardInputsI, forwardInputsP,
            forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
            forwardSpike, forwardSpikePrev, xedecay, xidecay, xpdecay, synInputBalanced,
            synInput, r, bias, p.wid, p.example_neurons, lastSpike,
            plusone, k, v, p.rng, noise, sig, P, Px, w0Index,
            w0Weights, nc0, stim, xtarg, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn,
            wpWeightOut, ncpIn, ncpOut, nothing, nothing)

        pcor = zeros(p.Ncells)
        for (index, ci) in enumerate(1:p.Ncells)
            xtarg_slice = @view xtarg[:,ci]
            xtotal_slice = @view xtotal[:,ci]
            pcor[index] = cor(xtarg_slice,xtotal_slice)
        end

        bnotnan = .!isnan.(pcor)
        println("cor = ", mean(pcor[bnotnan]),
                all(bnotnan) ? "" : string(" (", length(pcor)-count(bnotnan)," are NaN)"))
    end

    elapsed_time = time()-start_time
    println("elapsed time: ",elapsed_time, " sec")
    println("firing rate: ",mean(ns)/(dt/1000*p.Nsteps), " Hz")

    save(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$iloop.jld"),
         "wpWeightIn", collect(wpWeightIn))
    save(joinpath(parsed_args["data_dir"],"wpWeightOut-ckpt$iloop.jld"),
         "wpWeightOut", wpWeightOut)
    if (isnothing(parsed_args["save_checkpoints"]) && iloop>1) ||
       (!isnothing(parsed_args["save_checkpoints"]) && iloop % parsed_args["save_checkpoints"] != 1)
        rm(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$(iloop-1).jld"))
        rm(joinpath(parsed_args["data_dir"],"wpWeightOut-ckpt$(iloop-1).jld"))
    end
end # end loop over trainings

if !isnothing(parsed_args["monitor_resources_used"])
  sleep(60)
  close(chnl)
end
