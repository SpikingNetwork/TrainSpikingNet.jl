using LinearAlgebra, Random, JLD2, Statistics, CUDA, NNlib, NNlibCUDA, ArgParse, SymmetricFormats, BatchedBLAS

s = ArgParseSettings()

@add_arg_table! s begin
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

parsed_args = parse_args(s)

# --- load code --- #
include(joinpath(dirname(@__DIR__),"struct.jl"))
p = load(joinpath(parsed_args["data_dir"],"param.jld2"), "p")

kind=:train
include("convertWgtIn2Out.jl")
include("loop.jl")
include("rls.jl")
if parsed_args["correlation_interval"] <= parsed_args["nloops"]
    kind=:train_test
    include("loop.jl")
end

#----------- load initialization --------------#
w0Index = load(joinpath(parsed_args["data_dir"],"w0Index.jld2"), "w0Index");
w0Weights = load(joinpath(parsed_args["data_dir"],"w0Weights.jld2"), "w0Weights");
nc0 = load(joinpath(parsed_args["data_dir"],"nc0.jld2"), "nc0");
stim = load(joinpath(parsed_args["data_dir"],"stim.jld2"), "stim");
xtarg = load(joinpath(parsed_args["data_dir"],"xtarg.jld2"), "xtarg");
wpIndexIn = load(joinpath(parsed_args["data_dir"],"wpIndexIn.jld2"), "wpIndexIn");
wpIndexOut = load(joinpath(parsed_args["data_dir"],"wpIndexOut.jld2"), "wpIndexOut");
wpIndexConvert = load(joinpath(parsed_args["data_dir"],"wpIndexConvert.jld2"), "wpIndexConvert");
ffwdRate = load(joinpath(parsed_args["data_dir"],"ffwdRate.jld2"), "ffwdRate");
if isnothing(parsed_args["restore_from_checkpoint"])
    R=0
    wpWeightFfwd = load(joinpath(parsed_args["data_dir"],"wpWeightFfwd.jld2"), "wpWeightFfwd");
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn.jld2"), "wpWeightIn");
    Pinv_norm = load(joinpath(parsed_args["data_dir"],"P.jld2"), "P");
    P1 = p.PType(Array{Float64}(undef, p.Lexc+p.Linh+p.Lffwd, p.Lexc+p.Linh+p.Lffwd));
    P = Array{Float64}(undef, (size(p.PType==SymmetricPacked ? P1.tri : P1)..., p.Ncells));
    P .= p.PType==SymmetricPacked ? Pinv_norm.tri : Pinv_norm;
    if p.PPrecision<:Integer
        P .= round.(P .* p.PScale)
    end
else
    R = parsed_args["restore_from_checkpoint"]
    wpWeightFfwd = load(joinpath(parsed_args["data_dir"],"wpWeightFfwd-ckpt$R.jld2"), "wpWeightFfwd");
    wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld2"), "wpWeightIn");
    P = load(joinpath(parsed_args["data_dir"],"P-ckpt$R.jld2"), "P");
end;
wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells);
wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut);

rng = eval(p.rng_func["gpu"])
isnothing(p.seed) || Random.seed!(rng, p.seed)
save(joinpath(parsed_args["data_dir"],"rng-train.jld2"), "rng",rng)

#--- set up correlation matrix ---#
Px = wpIndexIn'; # neurons presynaptic to ci

# --- set up variables --- #
include("variables.jl")
Px = CuArray{p.IntPrecision}(Px);
P = CuArray{p.PPrecision}(P);
nc0 = CuArray{p.IntPrecision}(nc0);
stim = CuArray{p.FloatPrecision}(stim);
xtarg = CuArray{p.FloatPrecision}(xtarg);
w0Index = CuArray{p.IntPrecision}(w0Index);
w0Weights = CuArray{p.FloatPrecision}(w0Weights);
wpIndexIn = CuArray{p.IntPrecision}(wpIndexIn);
wpIndexConvert = CuArray{p.IntPrecision}(wpIndexConvert);
wpIndexOut = CuArray{p.IntPrecision}(wpIndexOut);
wpWeightFfwd = CuArray{p.FloatPrecision}(wpWeightFfwd);
wpWeightIn = CuArray{p.FloatPrecision}(wpWeightIn);
wpWeightOut = CuArray{p.FloatPrecision}(wpWeightOut);
ffwdRate = CuArray{p.FloatPrecision}(ffwdRate);

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
    sleep(parsed_args["monitor_resources_used"])
  end
end

if !isnothing(parsed_args["monitor_resources_used"])
  chnl = Channel(monitor_resources)
  sleep(60)
end

#----------- train the network --------------#
function make_symmetric(A::Array)
    for i=1:size(A,1), j=i+1:size(A,2)
        A[j,i,:] = A[i,j,:]
    end
    return A
end

maxcor = -Inf
for iloop = R.+(1:parsed_args["nloops"])
    println("Loop no. ",iloop) 

    start_time = time()

    if mod(iloop, parsed_args["correlation_interval"]) != 0

        loop_train(
            p.learn_every, p.stim_on, p.stim_off, p.train_time, p.dt,
            p.Nsteps, p.Ncells, p.L, nothing, p.Lexc+p.Linh, p.refrac, vre, invtauedecay,
            invtauidecay, invtaudecay_plastic, mu, thresh, tau, nothing,
            nothing, ns, nothing, ns_ffwd, forwardInputsE, forwardInputsI, forwardInputsP,
            forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
            forwardSpike, forwardSpikePrev, xedecay, xidecay, xpdecay,
            synInputBalanced, synInput, r, s, bias, nothing, nothing, lastSpike,
            bnotrefrac, bspike, plusone, minusone, PScale, raug, k, den, e, delta, v,
            rng, noise, rndFfwd, sig, P, Px, w0Index, w0Weights, nc0, stim, xtarg,
            wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut,
            nothing, nothing, ffwdRate)
    else
        _, _, _, _, xtotal, _ = loop_train_test(
            p.learn_every, p.stim_on, p.stim_off, p.train_time, p.dt,
            p.Nsteps, p.Ncells, p.L, nothing, p.Lexc+p.Linh, p.refrac, vre, invtauedecay,
            invtauidecay, invtaudecay_plastic, mu, thresh, tau, maxTimes,
            times, ns, times_ffwd, ns_ffwd, forwardInputsE, forwardInputsI, forwardInputsP,
            forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev,
            forwardSpike, forwardSpikePrev, xedecay, xidecay, xpdecay,
            synInputBalanced, synInput, r, s, bias, p.wid, p.example_neurons,
            lastSpike, bnotrefrac, bspike, plusone, minusone, PScale, raug, k, den, e,
            delta, v, rng, noise, rndFfwd, sig, P, Px, w0Index, w0Weights, nc0,
            stim, xtarg, wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn,
            wpWeightOut, nothing, nothing, ffwdRate)

        pcor = zeros(p.Ncells)
        for (index, ci) in enumerate(1:p.Ncells)
            xtarg_slice = convert(Array{Float64}, xtarg[:,ci])
            xtotal_slice = Array(xtotal[:,ci])
            pcor[index] = cor(xtarg_slice,xtotal_slice)
        end

        bnotnan = .!isnan.(pcor)
        thiscor = mean(pcor[bnotnan])
        println("correlation: ", thiscor,
                all(bnotnan) ? "" : string(" (", length(pcor)-count(bnotnan)," are NaN)"))
        if parsed_args["save_best_checkpoint"] && thiscor>maxcor && all(bnotnan)
            suffix = string("ckpt", iloop, "-cor", round(thiscor, digits=3))
            save(joinpath(parsed_args["data_dir"], "wpWeightFfwd-$suffix.jld2"),
                 "wpWeightFfwd", Array(wpWeightFfwd))
            save(joinpath(parsed_args["data_dir"], "wpWeightIn-$suffix.jld2"),
                 "wpWeightIn", Array(wpWeightIn))
            save(joinpath(parsed_args["data_dir"], "P-$suffix.jld2"),
                 "P", p.PType==Symmetric ? make_symmetric(Array(P)) : Array(P))
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
        save(joinpath(parsed_args["data_dir"],"wpWeightFfwd-ckpt$iloop.jld2"),
             "wpWeightFfwd", Array(wpWeightFfwd))
        save(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$iloop.jld2"),
             "wpWeightIn", Array(wpWeightIn))
        save(joinpath(parsed_args["data_dir"],"P-ckpt$iloop.jld2"),
             "P", p.PType==Symmetric ? make_symmetric(Array(P)) : Array(P))
    end

    elapsed_time = time()-start_time
    println("elapsed time: ",elapsed_time, " sec")
    println("firing rate: ",mean(ns)/(p.dt/1000*p.Nsteps), " Hz")
end

if !isnothing(parsed_args["monitor_resources_used"])
  sleep(60)
  close(chnl)
end
