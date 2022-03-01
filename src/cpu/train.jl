using LinearAlgebra, Random, JLD2, Statistics, ArgParse, SymmetricFormats

s = ArgParseSettings()

@add_arg_table! s begin
    "--nloops", "-n"
        help = "number of iterations to train"
        arg_type = Int
        default = 1
    "--performance_interval", "-p"
        help = "measure correlation every P training loops.  default is never"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "--save_checkpoints", "-c"
        help = "save the learned weights and covariance matrices every C training loops.  the default is after the last one"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "--restore_from_checkpoint", "-r"
        help = "continue training from checkpoint R.  default is to start from the beginning"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "--monitor_resources_used", "-m"
        help = "measure power, cores, and memory usage every R seconds.  default is never"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
end

parsed_args = parse_args(s)

BLAS.set_num_threads(1)

# --- load code --- #
include(joinpath(dirname(@__DIR__),"struct.jl"))
p = load(joinpath(parsed_args["data_dir"],"param.jld2"), "p")

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

#----------- load initialization --------------#
w0Index = load(joinpath(parsed_args["data_dir"],"w0Index.jld2"), "w0Index");
w0Weights = load(joinpath(parsed_args["data_dir"],"w0Weights.jld2"), "w0Weights");
nc0 = load(joinpath(parsed_args["data_dir"],"nc0.jld2"), "nc0");
ncpIn = load(joinpath(parsed_args["data_dir"],"ncpIn.jld2"), "ncpIn");
ncpOut = load(joinpath(parsed_args["data_dir"],"ncpOut.jld2"), "ncpOut");
stim = load(joinpath(parsed_args["data_dir"],"stim.jld2"), "stim");
xtarg = load(joinpath(parsed_args["data_dir"],"xtarg.jld2"), "xtarg");
wpIndexIn = load(joinpath(parsed_args["data_dir"],"wpIndexIn.jld2"), "wpIndexIn");
wpIndexOut = load(joinpath(parsed_args["data_dir"],"wpIndexOut.jld2"), "wpIndexOut");
wpIndexConvert = load(joinpath(parsed_args["data_dir"],"wpIndexConvert.jld2"), "wpIndexConvert");
if isnothing(parsed_args["restore_from_checkpoint"]);
    R=0
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn.jld2"), "wpWeightIn");
    Pinv_norm = load(joinpath(parsed_args["data_dir"],"P.jld2"), "P");
    P = Vector{p.PType}();
    for ci=1:p.Ncells
        push!(P, copy(Pinv_norm));
    end
else
    R = parsed_args["restore_from_checkpoint"];
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld2"), "wpWeightIn");
    P = load(joinpath(parsed_args["data_dir"],"P-ckpt$R.jld2"), "P");
end;
wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells);
wpWeightOut = convertWgtIn2Out(p.Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut);

rng = eval(p.rng_func["cpu"])
isnothing(p.seed) || Random.seed!(rng, p.seed)
save(joinpath(parsed_args["data_dir"],"rng-train.jld2"), "rng", rng)

# --- set up correlation matrix --- #
Px = Vector{Array{Int64,1}}();
for ci=1:p.Ncells
    push!(Px, wpIndexIn[ci,:]); # neurons presynaptic to ci
end

# --- set up variables --- #
include(joinpath(@__DIR__,"variables.jl"))
Px = Vector{Vector{p.IntPrecision}}(Px);
PType = typeof(p.PType(p.FloatPrecision.([1. 2; 3 4])));
P = Vector{PType}(P);
stim = Array{p.FloatPrecision}(stim);
xtarg = Array{p.FloatPrecision}(xtarg);
nc0 = Array{p.IntPrecision}(nc0);
ncpIn = Array{p.IntPrecision}(ncpIn);
ncpOut = Array{p.IntPrecision}(ncpOut);
w0Index = Array{p.IntPrecision}(w0Index);
w0Weights = Array{p.FloatPrecision}(w0Weights);
wpIndexIn = Array{p.IntPrecision}(wpIndexIn);
wpIndexConvert = Array{p.IntPrecision}(wpIndexConvert);
wpIndexOut = Array{p.IntPrecision}(wpIndexOut);
wpWeightIn = Array{p.FloatPrecision}(wpWeightIn);
wpWeightOut = Array{p.FloatPrecision}(wpWeightOut);

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
            synInputBalanced, synInput, r, bias, nothing, nothing,
            lastSpike, plusone, exactlyzero, k, v, rng, noise, sig, P,
            Px, w0Index, w0Weights, nc0, stim, xtarg, wpIndexIn, wpIndexOut,
            wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, nothing,
            nothing)
    else
        _, _, xtotal, _ = loop_train_test(
            p.learn_every, p.stim_on, p.stim_off, p.train_time, dt,
            p.Nsteps, p.Ncells, nothing, refrac, vre, invtauedecay,
            invtauidecay, invtaudecay_plastic, mu, thresh, invtau, maxTimes,
            times, ns, forwardInputsE, forwardInputsI, forwardInputsP,
            forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
            forwardSpike, forwardSpikePrev, xedecay, xidecay, xpdecay,
            synInputBalanced, synInput, r, bias, p.wid, p.example_neurons,
            lastSpike, plusone, exactlyzero, k, v, rng, noise, sig, P,
            Px, w0Index, w0Weights, nc0, stim, xtarg, wpIndexIn, wpIndexOut,
            wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, nothing,
            nothing)

        pcor = zeros(p.Ncells)
        for (index, ci) in enumerate(1:p.Ncells)
            xtarg_slice = @view xtarg[:,ci]
            xtotal_slice = @view xtotal[:,ci]
            pcor[index] = cor(xtarg_slice,xtotal_slice)
        end

        bnotnan = .!isnan.(pcor)
        println("correlation: ", mean(pcor[bnotnan]),
                all(bnotnan) ? "" : string(" (", length(pcor)-count(bnotnan)," are NaN)"))
    end

    if (isnothing(parsed_args["save_checkpoints"]) && iloop == R+parsed_args["nloops"]) ||
       (!isnothing(parsed_args["save_checkpoints"]) && iloop % parsed_args["save_checkpoints"] != 1)
        save(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$iloop.jld2"),
             "wpWeightIn", wpWeightIn)
        save(joinpath(parsed_args["data_dir"],"P-ckpt$iloop.jld2"), "P", P)
    end

    elapsed_time = time()-start_time
    println("elapsed time: ",elapsed_time, " sec")
    println("firing rate: ",mean(ns)/(dt/1000*p.Nsteps), " Hz")
end # end loop over trainings

if !isnothing(parsed_args["monitor_resources_used"])
  sleep(60)
  close(chnl)
end
