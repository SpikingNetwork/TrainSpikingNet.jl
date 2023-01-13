using Pkg;  Pkg.activate(dirname(dirname(@__DIR__)), io=devnull)

using LinearAlgebra, LinearAlgebra.BLAS, Random, JLD2, Statistics, ArgParse, SymmetricFormats

# --- define command line arguments --- #
aps = ArgParseSettings()

@add_arg_table! aps begin
    "--nloops", "-n"
        help = "number of iterations to train"
        arg_type = Int
        default = 1
    "--correlation_interval", "-c"
        help = "measure correlation every C training loops.  default is every loop"
        arg_type = Int
        default = 1
        range_tester = x->x>0
    "--save_best_checkpoint", "-s"
        help = "save the learned weights and covariance matrices with the highest measured correlation too.  default is to only save the last one"
        action = :store_true
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

parsed_args = parse_args(aps)

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
include("convertWgtIn2Out.jl")
include("loop.jl")
include("rls.jl")
if parsed_args["correlation_interval"] <= parsed_args["nloops"]
    kind=:train_test
    include("loop.jl")
end

# --- load initialization --- #
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
ffwdRate = load(joinpath(parsed_args["data_dir"],"ffwdRate.jld2"), "ffwdRate");
if isnothing(parsed_args["restore_from_checkpoint"]);
    R=0
    wpWeightFfwd = load(joinpath(parsed_args["data_dir"],"wpWeightFfwd.jld2"), "wpWeightFfwd");
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn.jld2"), "wpWeightIn");
    Pinv_norm = load(joinpath(parsed_args["data_dir"],"P.jld2"), "P");
    P = Vector{p.PType}();
    for ci=1:p.Ncells
        if p.PPrecision<:AbstractFloat
            push!(P, copy(Pinv_norm));
        else
            push!(P, round.(Pinv_norm * p.PScale));
        end
    end
else
    R = parsed_args["restore_from_checkpoint"];
    wpWeightFfwd = load(joinpath(parsed_args["data_dir"],"wpWeightFfwd-ckpt$R.jld2"), "wpWeightFfwd");
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld2"), "wpWeightIn");
    P = load(joinpath(parsed_args["data_dir"],"P-ckpt$R.jld2"), "P");
end;
wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells);
wpWeightOut = convertWgtIn2Out(p.Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut);

rng = eval(p.rng_func["cpu"])
isnothing(p.seed) || Random.seed!(rng, p.seed)
save(joinpath(parsed_args["data_dir"],"rng-train.jld2"), "rng", rng)

choose_task = eval(p.choose_task_func)
ntasks = size(xtarg,3)

# --- set up correlation matrix --- #
Px = Vector{Array{Int64,1}}();
for ci=1:p.Ncells
    push!(Px, wpIndexIn[ci,:]); # neurons presynaptic to ci
end

# --- set up variables --- #
include("variables.jl")
Px = Vector{Vector{p.IntPrecision}}(Px);
PType = typeof(p.PType(p.PPrecision.([1. 2; 3 4])));
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
wpWeightFfwd = Array{p.FloatPrecision}(wpWeightFfwd);
wpWeightOut = Array{p.FloatPrecision}(wpWeightOut);
ffwdRate = Array{p.FloatPrecision}(ffwdRate);

# --- monitor resources used --- #
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

# --- train the network --- #
maxcor = -Inf
for iloop = R.+(1:parsed_args["nloops"])
    itask = choose_task(iloop, ntasks)
    println("Loop no. ", iloop, ", task no. ", itask) 

    start_time = time()

    if mod(iloop, parsed_args["correlation_interval"]) != 0

        loop_train(itask,
            p.learn_every, p.stim_on, p.stim_off, p.train_time, p.dt,
            p.Nsteps, p.Ncells, nothing, p.Lexc+p.Linh, p.refrac, vre,
            invtauedecay, invtauidecay, invtaudecay_plastic, mu, thresh,
            tau, nothing, nothing, ns, nothing, ns_ffwd, forwardInputsE,
            forwardInputsI, forwardInputsP, forwardInputsEPrev,
            forwardInputsIPrev, forwardInputsPPrev, forwardSpike,
            forwardSpikePrev, xedecay, xidecay, xpdecay, synInputBalanced,
            synInput, r, s, bias, nothing, nothing, lastSpike, plusone,
            exactlyzero, PScale, raug, k, v, rng, noise, rndFfwd, sig, P,
            Px, w0Index, w0Weights, nc0, stim, xtarg, wpIndexIn, wpIndexOut,
            wpIndexConvert, wpWeightFfwd, wpWeightIn, wpWeightOut, ncpIn,
            ncpOut, nothing, nothing, ffwdRate)
    else
        _, _, _, _, xtotal, _, _, xplastic, _ = loop_train_test(itask,
            p.learn_every, p.stim_on, p.stim_off, p.train_time, p.dt,
            p.Nsteps, p.Ncells, nothing, p.Lexc+p.Linh, p.refrac,
            vre, invtauedecay, invtauidecay, invtaudecay_plastic,
            mu, thresh, tau, maxTimes, times, ns, times_ffwd,
            ns_ffwd, forwardInputsE, forwardInputsI, forwardInputsP,
            forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
            forwardSpike, forwardSpikePrev, xedecay, xidecay, xpdecay,
            synInputBalanced, synInput, r, s, bias, p.wid, p.example_neurons,
            lastSpike, plusone, exactlyzero, PScale, raug, k, v, rng, noise,
            rndFfwd, sig, P, Px, w0Index, w0Weights, nc0, stim, xtarg,
            wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightFfwd, wpWeightIn,
            wpWeightOut, ncpIn, ncpOut, nothing, nothing, ffwdRate)

        if p.correlation_var == :xtotal
            xlearned = xtotal
        elseif p.correlation_var == :xplastic
            xlearned = xplastic
        else
            error("invalid value for correlation_var parameter")
        end
        pcor = Array{Float64}(undef, p.Ncells)
        for ci in 1:p.Ncells
            xtarg_slice = @view xtarg[:,ci, itask]
            xlearned_slice = @view xlearned[:,ci]
            pcor[ci] = cor(xtarg_slice,xlearned_slice)
        end

        bnotnan = .!isnan.(pcor)
        thiscor = mean(pcor[bnotnan])
        println("correlation: ", thiscor,
                all(bnotnan) ? "" : string(" (", length(pcor)-count(bnotnan)," are NaN)"))

        if parsed_args["save_best_checkpoint"] && thiscor>maxcor && all(bnotnan)
            suffix = string("ckpt", iloop, "-cor", round(thiscor, digits=3))
            save(joinpath(parsed_args["data_dir"], "wpWeightIn-$suffix.jld2"),
                 "wpWeightIn", Array(wpWeightIn))
            save(joinpath(parsed_args["data_dir"], "wpWeightFfwd-$suffix.jld2"),
                 "wpWeightFfwd", Array(wpWeightFfwd))
            save(joinpath(parsed_args["data_dir"], "P-$suffix.jld2"), "P", P)
            if maxcor != -Inf
                for oldckptfile in filter(x -> !contains(x, string("ckpt", iloop)) &&
                                          contains(x, string("-cor", round(maxcor, digits=3))),
                                          readdir(parsed_args["data_dir"]))
                    rm(joinpath(parsed_args["data_dir"], oldckptfile))
                end
            end
            global maxcor = max(maxcor, thiscor)
        end
    end

    if iloop == R+parsed_args["nloops"]
        save(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$iloop.jld2"),
             "wpWeightIn", Array(wpWeightIn))
        save(joinpath(parsed_args["data_dir"],"wpWeightFfwd-ckpt$iloop.jld2"),
             "wpWeightFfwd", Array(wpWeightFfwd))
        save(joinpath(parsed_args["data_dir"],"P-ckpt$iloop.jld2"), "P", P)
    end


    elapsed_time = time()-start_time
    println("elapsed time: ",elapsed_time, " sec")
    println("firing rate: ",mean(ns)/(p.dt/1000*p.Nsteps), " Hz")
end # end loop over trainings

if !isnothing(parsed_args["monitor_resources_used"])
  sleep(60)
  close(chnl)
end
