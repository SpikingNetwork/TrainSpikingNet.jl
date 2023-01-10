using Pkg;  Pkg.activate(dirname(dirname(@__DIR__)))

using LinearAlgebra, Random, JLD2, CUDA, NNlib, NNlibCUDA, ArgParse

# --- define command line arguments --- #
import ArgParse: parse_item

function ArgParse.parse_item(::Type{Vector{Int}}, x::AbstractString)
    return eval(Meta.parse(x))
end

s = ArgParseSettings()

@add_arg_table! s begin
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

parsed_args = parse_args(s)

Threads.nthreads() < ndevices() && @warn "performance is best if no. threads is set to no. GPUs"
if Threads.nthreads() > ndevices()
    @error "no. threads cannot exceed no. GPUs"
    exit()
end

# --- load code --- #
include(joinpath(dirname(@__DIR__),"struct.jl"))
p = load(joinpath(parsed_args["data_dir"],"param.jld2"), "p")

include("convertWgtIn2Out.jl")
include("rls.jl")
kind=:test
include("loop.jl")

# --- load initialization --- #
nc0 = load(joinpath(parsed_args["data_dir"],"nc0.jld2"), "nc0")
stim = load(joinpath(parsed_args["data_dir"],"stim.jld2"), "stim")
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
wpWeightFfwd = load(joinpath(parsed_args["data_dir"],"wpWeightFfwd.jld2"), "wpWeightFfwd")
wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld2"))["wpWeightIn"]
wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells)
wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

# --- set up variables --- #
include("variables.jl")
nc0 = CuArray{p.IntPrecision}(nc0)
stim = CuArray{p.FloatPrecision}(stim);
w0Index = CuArray{p.IntPrecision}(w0Index);
w0Weights = CuArray{p.FloatPrecision}(w0Weights);
wpIndexOut = CuArray{p.IntPrecision}(wpIndexOut);
wpWeightOut = CuArray{p.FloatPrecision}(wpWeightOut);

rng = eval(p.rng_func["gpu"])
isnothing(p.seed) || Random.seed!(rng, p.seed)
save(joinpath(parsed_args["data_dir"],"rng-test.jld2"), "rng", rng)

# --- test the network --- #
ntrials = parsed_args["ntrials"]
ntasks = size(stim,3)
nss = Array{Any}(undef, ntrials, ntasks);
timess = Array{Any}(undef, ntrials, ntasks);
xtotals = Array{Any}(undef, ntrials, ntasks);
copy_rng = [typeof(rng)() for _=1:ndevices()];
isnothing(p.seed) || Random.seed!.(copy_rng, p.seed)
for var in [:times, :ns, :times_ffwd, :ns_ffwd, :stim, :nc0, :thresh, :tau,
            :w0Index, :w0Weights, :wpWeightFfwd, :wpIndexOut, :wpWeightOut, :mu,
            :forwardInputsE, :forwardInputsI, :forwardInputsP,
            :forwardInputsEPrev, :forwardInputsIPrev, :forwardInputsPPrev,
            :xedecay, :xidecay, :xpdecay, :synInputBalanced, :synInput,
            :bias, :lastSpike, :bnotrefrac, :bspike, :v, :noise, :sig]
  @eval (device!(0); tmp = Array($var))
  @eval $(Symbol("copy_",var)) = [(device!(idevice-1); CuArray($tmp)) for idevice=1:ndevices()];
end
if typeof(p.taudecay_plastic)<:AbstractArray
  device!(0); tmp = Array(invtaudecay_plastic)
  copy_invtaudecay_plastic = [(device!(idevice-1); CuArray(tmp)) for idevice=1:ndevices()];
end
synchronize()
Threads.@threads for itrial=1:ntrials
    idevice = Threads.threadid()
    device!(idevice-1)
    for itask = 1:ntasks
        t = @elapsed thisns, thistimes, _, _, thisxtotal, _ = loop_test(itask,
              p.learn_every, p.stim_on, p.stim_off, p.train_time, p.dt,
              p.Nsteps, p.Ncells, p.L, nothing, nothing, p.refrac, vre, invtauedecay,
              invtauidecay,
              typeof(p.taudecay_plastic)<:Number ? invtaudecay_plastic : copy_invtaudecay_plastic[idevice],
              copy_mu[idevice],
              copy_thresh[idevice],
              copy_tau[idevice],
              maxTimes,
              copy_times[idevice],
              copy_ns[idevice],
              copy_times_ffwd[idevice],
              copy_ns_ffwd[idevice],
              copy_forwardInputsE[idevice],
              copy_forwardInputsI[idevice],
              copy_forwardInputsP[idevice],
              copy_forwardInputsEPrev[idevice],
              copy_forwardInputsIPrev[idevice],
              copy_forwardInputsPPrev[idevice],
              nothing, nothing,
              copy_xedecay[idevice],
              copy_xidecay[idevice],
              copy_xpdecay[idevice],
              copy_synInputBalanced[idevice],
              copy_synInput[idevice],
              nothing, nothing,
              copy_bias[idevice],
              p.wid, p.example_neurons,
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
              copy_stim[idevice],
              nothing,
              copy_wpWeightFfwd[idevice],
              nothing,
              copy_wpIndexOut[idevice],
              nothing, nothing,
              copy_wpWeightOut[idevice],
              nothing);
        nss[itrial, itask] = Array(thisns[parsed_args["ineurons_to_test"]])
        timess[itrial, itask] = Array(thistimes[parsed_args["ineurons_to_test"],:])
        xtotals[itrial, itask] = Array(thisxtotal[:,parsed_args["ineurons_to_test"]])
        println("trial #", itrial, ", task #", itask, ": ",round(t, sigdigits=3), " sec")
    end
end

save(joinpath(parsed_args["data_dir"], "test.jld2"),
     "ineurons_to_test", parsed_args["ineurons_to_test"],
     "nss", nss, "timess", timess, "xtotals", xtotals)

parsed_args["no-plot"] || include(joinpath(dirname(@__DIR__),"plot.jl"))
