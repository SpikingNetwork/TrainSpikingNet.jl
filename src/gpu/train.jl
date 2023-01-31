using Pkg;  Pkg.activate(dirname(dirname(@__DIR__)), io=devnull)

using LinearAlgebra, Random, JLD2, Statistics, CUDA, NNlib, NNlibCUDA, ArgParse, SymmetricFormats, BatchedBLAS

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
        help = "measure power, cores, and memory usage every M seconds.  default is never"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
end

parsed_args = parse_args(aps)

# --- load code --- #
Param = load(joinpath(parsed_args["data_dir"],"param.jld2"), "param")

kind=:train
include(Param.cellModel_file)
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
X_stim = load(joinpath(parsed_args["data_dir"],"X_stim.jld2"), "X_stim");
utarg = load(joinpath(parsed_args["data_dir"],"utarg.jld2"), "utarg");
wpIndexIn = load(joinpath(parsed_args["data_dir"],"wpIndexIn.jld2"), "wpIndexIn");
wpIndexOut = load(joinpath(parsed_args["data_dir"],"wpIndexOut.jld2"), "wpIndexOut");
wpIndexConvert = load(joinpath(parsed_args["data_dir"],"wpIndexConvert.jld2"), "wpIndexConvert");
rateX = load(joinpath(parsed_args["data_dir"],"rateX.jld2"), "rateX");
if isnothing(parsed_args["restore_from_checkpoint"])
    R=0
    wpWeightX = load(joinpath(parsed_args["data_dir"],"wpWeightX.jld2"), "wpWeightX");
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn.jld2"), "wpWeightIn");
    Pinv_norm = load(joinpath(parsed_args["data_dir"],"P.jld2"), "P");
    PLtot = Param.Lexc + Param.Linh + Param.LX
    P1 = Param.PType(Array{Float64}(undef, PLtot, PLtot));
    P = Array{Float64}(undef, (size(Param.PType==SymmetricPacked ? P1.tri : P1)..., Param.Ncells));
    P .= Param.PType==SymmetricPacked ? Pinv_norm.tri : Pinv_norm;
    if Param.PPrecision<:Integer
        P .= round.(P .* Param.PScale)
    end
else
    R = parsed_args["restore_from_checkpoint"]
    wpWeightX = load(joinpath(parsed_args["data_dir"],"wpWeightX-ckpt$R.jld2"), "wpWeightX");
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld2"), "wpWeightIn");
    P = load(joinpath(parsed_args["data_dir"],"P-ckpt$R.jld2"), "P");
end;
wpWeightOut = zeros(maximum(wpIndexConvert), Param.Ncells);
wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut);

rng = eval(Param.rng_func.gpu)
isnothing(Param.seed) || Random.seed!(rng, Param.seed)
save(joinpath(parsed_args["data_dir"],"rng-train.jld2"), "rng",rng)

choose_task = eval(Param.choose_task_func)
ntasks = size(utarg,3)

# --- set up variables --- #
include("variables.jl")
P = CuArray{Param.PPrecision}(P);
nc0 = CuArray{Param.IntPrecision}(nc0);
X_stim = CuArray{Param.FloatPrecision}(X_stim);
utarg = CuArray{Param.FloatPrecision}(utarg);
w0Index = CuArray{Param.IntPrecision}(w0Index);
w0Weights = CuArray{Param.FloatPrecision}(w0Weights);
wpIndexIn = CuArray{Param.IntPrecision}(wpIndexIn);
wpIndexConvert = CuArray{Param.IntPrecision}(wpIndexConvert);
wpIndexOut = CuArray{Param.IntPrecision}(wpIndexOut);
wpWeightX = CuArray{Param.FloatPrecision}(wpWeightX);
wpWeightIn = CuArray{Param.FloatPrecision}(wpWeightIn);
wpWeightOut = CuArray{Param.FloatPrecision}(wpWeightOut);
rateX = CuArray{Param.FloatPrecision}(rateX);

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
    nvidiasmi = readlines(pipeline(`nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory --format=csv,noheader,nounits`))
    data = dropdims(sum(hcat([parse.(Float64, split(strip(x), ',')) for x in nvidiasmi]...),
                        dims=2), dims=2)
    println("total power used: ", strip(ipmitool[1]), " Watts\n",
            "CPU cores used by this process: ", strip(top[1]), "%\n",
            "CPU memory used by this process: ", strip(top[2]), "%\n",
            "GPU power used: ", data[1], " Watts\n",
            "GPU cores used: ", data[2], "%\n",
            "GPU memory used: ", data[3], "%")
    sleep(parsed_args["monitor_resources_used"])
  end
end

if !isnothing(parsed_args["monitor_resources_used"])
  chnl = Channel(monitor_resources)
  sleep(60)
end

# --- train the network --- #
function make_symmetric(A::Array)
    for i=1:size(A,1), j=i+1:size(A,2)
        A[j,i,:] = A[i,j,:]
    end
    return A
end

maxcor = -Inf
for iloop = R.+(1:parsed_args["nloops"])
    itask = choose_task(iloop, ntasks)
    println("Loop no. ", iloop, ", task no. ", itask) 

    start_time = time()

    if mod(iloop, parsed_args["correlation_interval"]) != 0

        loop_train(itask,
            Param.learn_every, Param.stim_on, Param.stim_off,
            Param.train_time, Param.dt, Param.Nsteps, Param.Ncells,
            nothing, Param.Lexc+Param.Linh, Param.LX, Param.refrac,
            learn_step, invtau_bale, invtau_bali, invtau_plas, X_bal,
            nothing, nothing, ns, nothing, nsX, inputsE,
            inputsI, inputsP, inputsEPrev, inputsIPrev, inputsPPrev,
            spikes, spikesPrev, spikesX, spikesXPrev, u_bale, u_bali,
            uX_plas, u_bal, u, r, rX, X, nothing, nothing,
            lastSpike, bnotrefrac, bspike, plusone, minusone, Param.PScale, raug,
            k, den, e, delta, v, rng, noise, rndX, sig, P, w0Index,
            w0Weights, nc0, X_stim, utarg, wpWeightX, wpIndexIn, wpIndexOut,
            wpIndexConvert, wpWeightIn, wpWeightOut, rateX,
            cellModel_args)
    else
        _, _, _, _, utotal, _, _, uplastic, _ = loop_train_test(itask,
            Param.learn_every, Param.stim_on, Param.stim_off,
            Param.train_time, Param.dt, Param.Nsteps, Param.Ncells,
            nothing, Param.Lexc+Param.Linh, Param.LX, Param.refrac,
            learn_step, invtau_bale, invtau_bali, invtau_plas, X_bal,
            maxTimes, times, ns, timesX, nsX, inputsE, inputsI,
            inputsP, inputsEPrev, inputsIPrev, inputsPPrev, spikes,
            spikesPrev, spikesX, spikesXPrev, u_bale, u_bali,
            uX_plas, u_bal, u, r, rX, X, Param.wid,
            Param.example_neurons, lastSpike, bnotrefrac, bspike, plusone,
            minusone, Param.PScale, raug, k, den, e, delta, v, rng, noise, rndX,
            sig, P, w0Index, w0Weights, nc0, X_stim, utarg, wpWeightX,
            wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut,
            rateX, cellModel_args)

        if Param.correlation_var == :utotal
            ulearned = utotal
        elseif Param.correlation_var == :uplastic
            ulearned = uplastic
        else
            error("invalid value for correlation_var parameter")
        end
        pcor = Array{Float64}(undef, Param.Ncells)
        for ci in 1:Param.Ncells
            utarg_slice = convert(Array{Float64}, utarg[:,ci, itask])
            ulearned_slice = Array(ulearned[:,ci])
            pcor[ci] = cor(utarg_slice, ulearned_slice)
        end

        bnotnan = .!isnan.(pcor)
        thiscor = mean(pcor[bnotnan])
        println("correlation: ", thiscor,
                all(bnotnan) ? "" : string(" (", length(pcor)-count(bnotnan)," are NaN)"))

        if parsed_args["save_best_checkpoint"] && thiscor>maxcor && all(bnotnan)
            suffix = string("ckpt", iloop, "-cor", round(thiscor, digits=3))
            save(joinpath(parsed_args["data_dir"], "wpWeightX-$suffix.jld2"),
                 "wpWeightX", Array(wpWeightX))
            save(joinpath(parsed_args["data_dir"], "wpWeightIn-$suffix.jld2"),
                 "wpWeightIn", Array(wpWeightIn))
            save(joinpath(parsed_args["data_dir"], "P-$suffix.jld2"),
                 "P", Param.PType==Symmetric ? make_symmetric(Array(P)) : Array(P))
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
        save(joinpath(parsed_args["data_dir"],"wpWeightX-ckpt$iloop.jld2"),
             "wpWeightX", Array(wpWeightX))
        save(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$iloop.jld2"),
             "wpWeightIn", Array(wpWeightIn))
        save(joinpath(parsed_args["data_dir"],"P-ckpt$iloop.jld2"),
             "P", Param.PType==Symmetric ? make_symmetric(Array(P)) : Array(P))
    end

    elapsed_time = time()-start_time
    println("elapsed time: ", elapsed_time, " sec")
    println("firing rate: ", mean(ns) / (Param.dt/1000*Param.Nsteps), " Hz")
end

if !isnothing(parsed_args["monitor_resources_used"])
  sleep(60)
  close(chnl)
end
