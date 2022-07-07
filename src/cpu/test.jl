using LinearAlgebra, Random, JLD2, ArgParse

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
    "--no-plot"
        help = "just save to JLD2 file"
        action = :store_true
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
  quote $(esc(loop)); end
end

include("convertWgtIn2Out.jl")
include("rls.jl")
kind=:test
include("loop.jl")

#----------- load initialization --------------#
nc0 = load(joinpath(parsed_args["data_dir"],"nc0.jld2"), "nc0")
ncpIn = load(joinpath(parsed_args["data_dir"],"ncpIn.jld2"), "ncpIn")
ncpOut = load(joinpath(parsed_args["data_dir"],"ncpOut.jld2"), "ncpOut")
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
wpWeightFfwd = load(joinpath(parsed_args["data_dir"],"wpWeightFfwd-ckpt$R.jld2"), "wpWeightFfwd");
wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld2"), "wpWeightIn")
wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells)
wpWeightOut = convertWgtIn2Out(p.Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

rng = eval(p.rng_func["cpu"])
isnothing(p.seed) || Random.seed!(rng, p.seed)
save(joinpath(parsed_args["data_dir"],"rng-test.jld2"), "rng", rng)

# --- set up variables --- #
include("variables.jl")
stim = Array{p.FloatPrecision}(stim);
nc0 = Array{p.IntPrecision}(nc0)
ncpOut = Array{p.IntPrecision}(ncpOut);
w0Index = Array{p.IntPrecision}(w0Index);
w0Weights = Array{p.FloatPrecision}(w0Weights);
wpIndexOut = Array{p.IntPrecision}(wpIndexOut);
wpWeightOut = Array{p.FloatPrecision}(wpWeightOut);

#----------- test the network --------------#
nss = Vector{Any}(undef, parsed_args["ntrials"]);
timess = Vector{Any}(undef, parsed_args["ntrials"]);
xtotals = Vector{Any}(undef, parsed_args["ntrials"]);
copy_rng = [typeof(rng)() for _=1:Threads.nthreads()];
isnothing(p.seed) || Random.seed!.(copy_rng, p.seed)
for var in [:times, :ns, :times_ffwd, :ns_ffwd,
            :forwardInputsE, :forwardInputsI, :forwardInputsP,
            :forwardInputsEPrev, :forwardInputsIPrev, :forwardInputsPPrev,
            :xedecay, :xidecay, :xpdecay, :synInputBalanced, :synInput,
            :bias, :lastSpike, :v, :noise]
  @eval $(Symbol("copy_",var)) = [deepcopy($var) for _=1:Threads.nthreads()];
end
Threads.@threads for itrial=1:parsed_args["ntrials"]
    t = @elapsed thisns, thistimes, _, _, thisxtotal, _ = loop_test(
          p.learn_every, p.stim_on, p.stim_off, p.train_time, p.dt,
          p.Nsteps, p.Ncells, nothing, nothing, p.refrac, vre, invtauedecay,
          invtauidecay, invtaudecay_plastic, mu, thresh, tau, maxTimes,
          copy_times[Threads.threadid()],
          copy_ns[Threads.threadid()],
          copy_times_ffwd[Threads.threadid()],
          copy_ns_ffwd[Threads.threadid()],
          copy_forwardInputsE[Threads.threadid()],
          copy_forwardInputsI[Threads.threadid()],
          copy_forwardInputsP[Threads.threadid()],
          copy_forwardInputsEPrev[Threads.threadid()],
          copy_forwardInputsIPrev[Threads.threadid()],
          copy_forwardInputsPPrev[Threads.threadid()],
          nothing, nothing,
          copy_xedecay[Threads.threadid()],
          copy_xidecay[Threads.threadid()],
          copy_xpdecay[Threads.threadid()],
          copy_synInputBalanced[Threads.threadid()],
          copy_synInput[Threads.threadid()],
          nothing, nothing,
          copy_bias[Threads.threadid()],
          p.wid, p.example_neurons,
          copy_lastSpike[Threads.threadid()],
          nothing, nothing, nothing, nothing, nothing,
          copy_v[Threads.threadid()],
          copy_rng[Threads.threadid()],
          copy_noise[Threads.threadid()],
          nothing, sig, nothing, nothing, w0Index, w0Weights, nc0, stim, nothing,
          nothing, wpIndexOut, nothing, nothing, nothing, wpWeightOut, nothing,
          ncpOut, nothing, nothing, nothing)
    nss[itrial] = thisns[parsed_args["ineurons_to_plot"]]
    timess[itrial] = thistimes[parsed_args["ineurons_to_plot"],:]
    xtotals[itrial] = thisxtotal[:,parsed_args["ineurons_to_plot"]]
    println("trial #", itrial, ", ", round(t, sigdigits=3), " sec")
end

save(joinpath(parsed_args["data_dir"],"test.jld2"),
     "ineurons_to_plot", parsed_args["ineurons_to_plot"],
     "nss", nss, "timess", timess, "xtotals", xtotals)

parsed_args["no-plot"] || include(joinpath(dirname(@__DIR__),"plot.jl"))
