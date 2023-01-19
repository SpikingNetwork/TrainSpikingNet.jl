using Pkg;  Pkg.activate(dirname(dirname(@__DIR__)), io=devnull)

using LinearAlgebra, Random, JLD2, CUDA, NNlib, NNlibCUDA, ArgParse

# --- define command line arguments --- #
import ArgParse: parse_item

function ArgParse.parse_item(::Type{Vector{Int}}, x::AbstractString)
    return eval(Meta.parse(x))
end

aps = ArgParseSettings()

@add_arg_table! aps begin
    "--ntrials", "-n"
        help = "number of repeated trials to average over"
        arg_type = Int
        default = 1
        range_tester = x->x>0
    "--ineurons_to_test", "-i"
        help = "which neurons to test"
        arg_type = Vector{Int}
        default = collect(1:16)
        range_tester = x->all(x.>0)
    "--restore_from_checkpoint", "-r"
        help = "use checkpoint R.  default is to use the last one"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "--no-plot"
        help = "just save to JLD2 file"
        action = :store_true
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
end

parsed_args = parse_args(aps)

Threads.nthreads() < ndevices() && @warn "performance is best if no. threads is set to no. GPUs"
if Threads.nthreads() > ndevices()
    @error "no. threads cannot exceed no. GPUs"
    exit()
end

# --- load code --- #
Param = load(joinpath(parsed_args["data_dir"],"param.jld2"), "param")

include("convertWgtIn2Out.jl")
include("rls.jl")
kind=:test
include("loop.jl")

# --- load initialization --- #
nc0 = load(joinpath(parsed_args["data_dir"],"nc0.jld2"), "nc0")
X_stim = load(joinpath(parsed_args["data_dir"],"X_stim.jld2"), "X_stim")
w0Index = load(joinpath(parsed_args["data_dir"],"w0Index.jld2"), "w0Index")
w0Weights = load(joinpath(parsed_args["data_dir"],"w0Weights.jld2"), "w0Weights")
wpIndexIn = load(joinpath(parsed_args["data_dir"],"wpIndexIn.jld2"), "wpIndexIn")
wpIndexOut = load(joinpath(parsed_args["data_dir"],"wpIndexOut.jld2"), "wpIndexOut")
wpIndexConvert = load(joinpath(parsed_args["data_dir"],"wpIndexConvert.jld2"), "wpIndexConvert")
if isnothing(parsed_args["restore_from_checkpoint"])
    R = maximum([parse(Int, m.captures[1])
                 for m in match.(r"ckpt([0-9]+)\.jld2",
                                 filter(startswith("wpWeightIn-ckpt"),
                                        readdir(parsed_args["data_dir"])))])
else
    R = parsed_args["restore_from_checkpoint"]
end
wpWeightX = load(joinpath(parsed_args["data_dir"],"wpWeightX.jld2"), "wpWeightX")
wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld2"))["wpWeightIn"]
wpWeightOut = zeros(maximum(wpIndexConvert), Param.Ncells)
wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

# --- set up variables --- #
include("variables.jl")
nc0 = CuArray{Param.IntPrecision}(nc0)
X_stim = CuArray{Param.FloatPrecision}(X_stim);
w0Index = CuArray{Param.IntPrecision}(w0Index);
w0Weights = CuArray{Param.FloatPrecision}(w0Weights);
wpIndexOut = CuArray{Param.IntPrecision}(wpIndexOut);
wpWeightOut = CuArray{Param.FloatPrecision}(wpWeightOut);

rng = eval(Param.rng_func["gpu"])
isnothing(Param.seed) || Random.seed!(rng, Param.seed)
save(joinpath(parsed_args["data_dir"],"rng-test.jld2"), "rng", rng)

# --- test the network --- #
ntrials = parsed_args["ntrials"]
ntasks = size(X_stim,3)
nss = Array{Any}(undef, ntrials, ntasks);
timess = Array{Any}(undef, ntrials, ntasks);
utotals = Array{Any}(undef, ntrials, ntasks);
copy_rng = [typeof(rng)() for _=1:ndevices()];
isnothing(Param.seed) || Random.seed!.(copy_rng, Param.seed)
for var in [:times, :ns, :timesX, :nsX, :X_stim, :nc0, :thresh, :tau_mem,
            :w0Index, :w0Weights, :wpWeightX, :wpIndexOut, :wpWeightOut, :X_bal,
            :inputsE, :inputsI, :inputsP, :inputsEPrev, :inputsIPrev, :inputsPPrev,
            :u_bale, :u_bali, :uX_plas, :u_bal, :u,
            :X, :lastSpike, :bnotrefrac, :bspike, :v, :noise, :sig]
  @eval (device!(0); tmp = Array($var))
  @eval $(Symbol("copy_",var)) = [(device!(idevice-1); CuArray($tmp)) for idevice=1:ndevices()];
end
if typeof(Param.tau_plas)<:AbstractArray
  device!(0); tmp = Array(invtau_plas)
  copy_invtau_plas = [(device!(idevice-1); CuArray(tmp)) for idevice=1:ndevices()];
end
synchronize()
Threads.@threads for itrial=1:ntrials
    idevice = Threads.threadid()
    device!(idevice-1)
    for itask = 1:ntasks
        t = @elapsed thisns, thistimes, _, _, thisutotal, _ = loop_test(itask,
              Param.learn_every, Param.stim_on, Param.stim_off,
              Param.train_time, Param.dt, Param.Nsteps, Param.Ncells,
              nothing, nothing, Param.LX, Param.refrac, vre, invtau_bale,
              invtau_bali,
              typeof(Param.tau_plas)<:Number ? invtau_plas : copy_invtau_plas[idevice],
              copy_X_bal[idevice],
              copy_thresh[idevice],
              copy_tau_mem[idevice],
              maxTimes,
              copy_times[idevice],
              copy_ns[idevice],
              copy_timesX[idevice],
              copy_nsX[idevice],
              copy_inputsE[idevice],
              copy_inputsI[idevice],
              copy_inputsP[idevice],
              copy_inputsEPrev[idevice],
              copy_inputsIPrev[idevice],
              copy_inputsPPrev[idevice],
              nothing, nothing, nothing, nothing,
              copy_u_bale[idevice],
              copy_u_bali[idevice],
              copy_uX_plas[idevice],
              copy_u_bal[idevice],
              copy_u[idevice],
              nothing, nothing,
              copy_X[idevice],
              Param.wid, Param.example_neurons,
              copy_lastSpike[idevice],
              copy_bnotrefrac[idevice],
              copy_bspike[idevice],
              plusone,
              nothing, nothing, nothing, nothing, nothing, nothing, nothing,
              copy_v[idevice],
              copy_rng[idevice],
              copy_noise[idevice],
              nothing,
              copy_sig[idevice],
              nothing, nothing,
              copy_w0Index[idevice],
              copy_w0Weights[idevice],
              copy_nc0[idevice],
              copy_X_stim[idevice],
              nothing,
              copy_wpWeightX[idevice],
              nothing,
              copy_wpIndexOut[idevice],
              nothing, nothing,
              copy_wpWeightOut[idevice],
              nothing);
        nss[itrial, itask] = Array(thisns[parsed_args["ineurons_to_test"]])
        timess[itrial, itask] = Array(thistimes[parsed_args["ineurons_to_test"],:])
        utotals[itrial, itask] = Array(thisutotal[:,parsed_args["ineurons_to_test"]])
        println("trial #", itrial, ", task #", itask, ": ",round(t, sigdigits=3), " sec")
    end
end

save(joinpath(parsed_args["data_dir"], "test.jld2"),
     "ineurons_to_test", parsed_args["ineurons_to_test"],
     "nss", nss, "timess", timess, "utotals", utotals)

parsed_args["no-plot"] || run(`$(Base.julia_cmd())
                               $(joinpath(@__DIR__, "..", "plot.jl"))
                               -i $(repr(parsed_args["ineurons_to_test"]))
                               $(joinpath(parsed_args["data_dir"], "test.jld2"))`)
