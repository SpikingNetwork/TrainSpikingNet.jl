using LinearAlgebra, Random, JLD, CUDA, NNlib, NNlibCUDA, ArgParse

Threads.nthreads() < ndevices() && @warn "performance is best if no. threads is set to no. GPUs"
if Threads.nthreads() > ndevices()
    @error "no. threads cannot exceed no. GPUs"
    exit()
end

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
    "--ineurons_to_plot", "-i"
        help = "which neurons to plot"
        arg_type = Vector{Int}
        default = collect(1:16)
        range_tester = x->all(x.>0)
    "--restore_from_checkpoint", "-r"
        help = "use checkpoint R.  default is to use the last one"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
        metavar = "R"
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
end

parsed_args = parse_args(s)

# --- load code --- #
include(joinpath(dirname(@__DIR__),"struct.jl"))
p = load(joinpath(parsed_args["data_dir"],"p.jld"))["p"]

include(joinpath(@__DIR__,"convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"rls.jl"))
kind=:test
include(joinpath(@__DIR__,"loop.jl"))
include(joinpath(@__DIR__,"funRollingAvg.jl"))

#----------- load initialization --------------#
nc0 = load(joinpath(parsed_args["data_dir"],"nc0.jld"))["nc0"]
stim = load(joinpath(parsed_args["data_dir"],"stim.jld"))["stim"]
w0Index = load(joinpath(parsed_args["data_dir"],"w0Index.jld"))["w0Index"]
w0Weights = load(joinpath(parsed_args["data_dir"],"w0Weights.jld"))["w0Weights"]
wpIndexIn = load(joinpath(parsed_args["data_dir"],"wpIndexIn.jld"))["wpIndexIn"]
wpIndexOut = load(joinpath(parsed_args["data_dir"],"wpIndexOut.jld"))["wpIndexOut"]
wpIndexConvert = load(joinpath(parsed_args["data_dir"],"wpIndexConvert.jld"))["wpIndexConvert"]
if isnothing(parsed_args["restore_from_checkpoint"])
    R = maximum([parse(Int, m.captures[1])
                 for m in match.(r"ckpt([0-9]+)\.jld",
                                 filter(startswith("wpWeightIn-ckpt"),
                                        readdir(parsed_args["data_dir"])))])
else
    R = parsed_args["restore_from_checkpoint"]
end
wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld"))["wpWeightIn"]
wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells)
wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

# --- set up variables --- #
include(joinpath(@__DIR__,"variables.jl"))
nc0 = CuArray{p.IntPrecision}(nc0)
stim = CuArray{p.FloatPrecision}(stim);
w0Index = CuArray{p.IntPrecision}(w0Index);
w0Weights = CuArray{p.FloatPrecision}(w0Weights);
wpIndexOut = CuArray{p.IntPrecision}(wpIndexOut);
wpWeightOut = CuArray{p.FloatPrecision}(wpWeightOut);

#----------- test the network --------------#
nss = Vector{Any}(undef, parsed_args["ntrials"]);
timess = Vector{Any}(undef, parsed_args["ntrials"]);
xtotals = Vector{Any}(undef, parsed_args["ntrials"]);
copy_rng = [typeof(p.rng)() for _=1:ndevices()];
isnothing(p.seed) || Random.seed!.(copy_rng, p.seed)
for var in [:times, :ns, :stim, :nc0, :thresh, :invtau,
            :w0Index, :w0Weights, :wpIndexOut, :wpWeightOut,
            :mu, :invtaudecay_plastic,
            :forwardInputsE, :forwardInputsI, :forwardInputsP,
            :forwardInputsEPrev, :forwardInputsIPrev, :forwardInputsPPrev,
            :xedecay, :xidecay, :xpdecay, :synInputBalanced, :synInput,
            :bias, :lastSpike, :bnotrefrac, :bspike, :v, :noise, :sig]
  @eval (device!(0); tmp = Array($var))
  @eval $(Symbol("copy_",var)) = [(device!(idevice-1); CuArray($tmp)) for idevice=1:ndevices()];
end
synchronize()
Threads.@threads for itrial=1:parsed_args["ntrials"]
    idevice = Threads.threadid()
    device!(idevice-1)
    fill(copy_times[idevice], 0);
    fill(copy_ns[idevice], 0);
    t = @elapsed thisns, thistimes, thisxtotal, _ = loop_test(
          p.learn_every, p.stim_on, p.stim_off, p.train_time, dt,
          p.Nsteps, p.Ncells, p.L, nothing, refrac, vre, invtauedecay,
          invtauidecay,
          copy_invtaudecay_plastic[idevice],
          copy_mu[idevice],
          copy_thresh[idevice],
          copy_invtau[idevice],
          maxTimes,
          copy_times[idevice],
          copy_ns[idevice],
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
          nothing,
          copy_bias[idevice],
          p.wid, p.example_neurons,
          copy_lastSpike[idevice],
          copy_bnotrefrac[idevice],
          copy_bspike[idevice],
          nothing, nothing, nothing, nothing, nothing, nothing,
          copy_v[idevice],
          copy_rng[idevice],
          copy_noise[idevice],
          copy_sig[idevice],
          nothing, nothing,
          copy_w0Index[idevice],
          copy_w0Weights[idevice],
          copy_nc0[idevice],
          copy_stim[idevice],
          nothing, nothing,
          copy_wpIndexOut[idevice],
          nothing, nothing,
          copy_wpWeightOut[idevice],
          nothing, nothing);
    nss[itrial] = Array(thisns[parsed_args["ineurons_to_plot"]])
    timess[itrial] = Array(thistimes[parsed_args["ineurons_to_plot"],:])
    xtotals[itrial] = Array(thisxtotal[:,parsed_args["ineurons_to_plot"]])
    @info string("trial #", itrial, ", ", round(t, sigdigits=3), " sec")
end

save(joinpath(parsed_args["data_dir"],"test.jld"),
     "ineurons_to_plot", parsed_args["ineurons_to_plot"],
     "nss", nss, "timess", timess, "xtotals", xtotals)

include(joinpath(dirname(@__DIR__),"plot.jl"))
