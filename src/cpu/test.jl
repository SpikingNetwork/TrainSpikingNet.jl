using LinearAlgebra, Random, JLD, ArgParse

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
wpIndexOut = load(joinpath(parsed_args["data_dir"],"wpIndexOut.jld"))["wpIndexOut"]
wpWeightOut = load(joinpath(parsed_args["data_dir"],"wpWeightOut-trained.jld"))["wpWeightOut"]
ncpOut = load(joinpath(parsed_args["data_dir"],"ncpOut.jld"))["ncpOut"]

# --- load code --- #
macro maybethread(loop)
  quote $(esc(loop)); end
end

include(joinpath(@__DIR__,"convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"rls.jl"))
kind=:test
include(joinpath(@__DIR__,"loop.jl"))

# --- set up variables --- #
include(joinpath(@__DIR__,"variables.jl"))
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
copy_rng = [typeof(p.rng)() for _=1:Threads.nthreads()];
isnothing(p.seed) || Random.seed!.(copy_rng, p.seed)
for var in [:times, :ns,
            :forwardInputsE, :forwardInputsI, :forwardInputsP,
            :forwardInputsEPrev, :forwardInputsIPrev, :forwardInputsPPrev,
            :xedecay, :xidecay, :xpdecay, :synInputBalanced, :synInput,
            :bias, :lastSpike, :v, :noise]
  @eval $(Symbol("copy_",var)) = [deepcopy($var) for _=1:Threads.nthreads()];
end
Threads.@threads for itrial=1:parsed_args["ntrials"]
    fill(copy_times[Threads.threadid()], 0);
    fill(copy_ns[Threads.threadid()], 0);
    t = @elapsed thisns, thistimes, thisxtotal, _ = loop_test(
          p.learn_every, p.stim_on, p.stim_off, p.train_time, dt,
          p.Nsteps, p.Ncells, nothing, refrac, vre, invtauedecay,
          invtauidecay, invtaudecay_plastic, mu, thresh, invtau, maxTimes,
          copy_times[Threads.threadid()],
          copy_ns[Threads.threadid()],
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
          nothing,
          copy_bias[Threads.threadid()],
          p.wid, p.example_neurons,
          copy_lastSpike[Threads.threadid()],
          nothing, nothing,
          copy_v[Threads.threadid()],
          copy_rng[Threads.threadid()],
          copy_noise[Threads.threadid()],
          sig, nothing, nothing, w0Index, w0Weights, nc0, stim, nothing,
          nothing, wpIndexOut, nothing, nothing, wpWeightOut, nothing,
          ncpOut, nothing, nothing)
    nss[itrial] = thisns[parsed_args["ineurons_to_plot"]]
    timess[itrial] = thistimes[parsed_args["ineurons_to_plot"],:]
    xtotals[itrial] = thisxtotal[:,parsed_args["ineurons_to_plot"]]
    @info string("trial #", itrial, ", ", round(t, sigdigits=3), " sec")
end

save(joinpath(parsed_args["data_dir"],"test.jld"),
     "ineurons_to_plot", parsed_args["ineurons_to_plot"],
     "nss", nss, "timess", timess, "xtotals", xtotals)

include(joinpath(dirname(@__DIR__),"plot.jl"))
