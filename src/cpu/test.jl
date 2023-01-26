using Pkg;  Pkg.activate(dirname(dirname(@__DIR__)), io=devnull)

using LinearAlgebra, LinearAlgebra.BLAS, Random, JLD2, ArgParse

println(BLAS.get_config())

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

BLAS.set_num_threads(1)

# --- load code --- #
Param = load(joinpath(parsed_args["data_dir"],"param.jld2"), "param")

macro maybethread(loop)
  quote $(esc(loop)); end
end

include("convertWgtIn2Out.jl")
include("rls.jl")
kind=:test
include("loop.jl")

# --- load initialization --- #
nc0 = load(joinpath(parsed_args["data_dir"],"nc0.jld2"), "nc0")
ncpIn = load(joinpath(parsed_args["data_dir"],"ncpIn.jld2"), "ncpIn")
ncpOut = load(joinpath(parsed_args["data_dir"],"ncpOut.jld2"), "ncpOut")
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
wpWeightX = load(joinpath(parsed_args["data_dir"],"wpWeightX-ckpt$R.jld2"), "wpWeightX");
wpWeightIn = load(joinpath(parsed_args["data_dir"],"wpWeightIn-ckpt$R.jld2"), "wpWeightIn")
wpWeightOut = zeros(maximum(wpIndexConvert), Param.Ncells)
wpWeightOut = convertWgtIn2Out(Param.Ncells, ncpIn,
                               wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut)

rng = eval(Param.rng_func["cpu"])
isnothing(Param.seed) || Random.seed!(rng, Param.seed)
save(joinpath(parsed_args["data_dir"],"rng-test.jld2"), "rng", rng)

# --- set up variables --- #
include("variables.jl")
X_stim = Array{Param.FloatPrecision}(X_stim);
nc0 = Array{Param.IntPrecision}(nc0)
ncpOut = Array{Param.IntPrecision}(ncpOut);
w0Index = Array{Param.IntPrecision}(w0Index);
w0Weights = Array{Param.FloatPrecision}(w0Weights);
wpIndexOut = Array{Param.IntPrecision}(wpIndexOut);
wpWeightOut = Array{Param.FloatPrecision}(wpWeightOut);

# --- test the network --- #
ntrials = parsed_args["ntrials"]
ntasks = size(X_stim,3)
nss = Array{Any}(undef, ntrials, ntasks);
timess = Array{Any}(undef, ntrials, ntasks);
utotals = Array{Any}(undef, ntrials, ntasks);
copy_rng = [typeof(rng)() for _=1:Threads.nthreads()];
isnothing(Param.seed) || Random.seed!.(copy_rng, Param.seed)
for var in [:times, :ns, :timesX, :nsX,
            :inputsE, :inputsI, :inputsP, :inputsEPrev, :inputsIPrev, :inputsPPrev,
            :u_bale, :u_bali, :uX_plas, :u_bal, :u,
            :X, :lastSpike, :v, :noise]
  @eval $(Symbol("copy_",var)) = [deepcopy($var) for _=1:Threads.nthreads()];
end
Threads.@threads for itrial=1:ntrials
    for itask = 1:ntasks
        t = @elapsed thisns, thistimes, _, _, thisutotal, _ = loop_test(itask,
              Param.learn_every, Param.stim_on, Param.stim_off,
              Param.train_time, Param.dt, Param.Nsteps, Param.Ncells,
              nothing, nothing, Param.LX, Param.refrac, vre, invtau_bale,
              invtau_bali, invtau_plas, X_bal, thresh, tau_mem, maxTimes,
              copy_times[Threads.threadid()],
              copy_ns[Threads.threadid()],
              copy_timesX[Threads.threadid()],
              copy_nsX[Threads.threadid()],
              copy_inputsE[Threads.threadid()],
              copy_inputsI[Threads.threadid()],
              copy_inputsP[Threads.threadid()],
              copy_inputsEPrev[Threads.threadid()],
              copy_inputsIPrev[Threads.threadid()],
              copy_inputsPPrev[Threads.threadid()],
              nothing, nothing, nothing, nothing,
              copy_u_bale[Threads.threadid()],
              copy_u_bali[Threads.threadid()],
              copy_uX_plas[Threads.threadid()],
              copy_u_bal[Threads.threadid()],
              copy_u[Threads.threadid()],
              nothing, nothing,
              copy_X[Threads.threadid()],
              Param.wid, Param.example_neurons,
              copy_lastSpike[Threads.threadid()],
              nothing, nothing, nothing, nothing, nothing,
              copy_v[Threads.threadid()],
              copy_rng[Threads.threadid()],
              copy_noise[Threads.threadid()],
              nothing, sig, nothing, w0Index, w0Weights, nc0, X_stim, nothing,
              nothing, wpIndexOut, nothing, nothing, nothing, wpWeightOut, nothing,
              ncpOut, nothing, nothing, nothing)
        nss[itrial, itask] = thisns[parsed_args["ineurons_to_test"]]
        timess[itrial, itask] = thistimes[parsed_args["ineurons_to_test"],:]
        utotals[itrial, itask] = thisutotal[:,parsed_args["ineurons_to_test"]]
        println("trial #", itrial, ", task #", itask, ": ",round(t, sigdigits=3), " sec")
    end
end

save(joinpath(parsed_args["data_dir"],"test.jld2"),
     "ineurons_to_test", parsed_args["ineurons_to_test"],
     "nss", nss, "timess", timess, "utotals", utotals)

parsed_args["no-plot"] || run(`$(Base.julia_cmd())
                               $(joinpath(@__DIR__, "..", "plot.jl"))
                               -i $(repr(parsed_args["ineurons_to_test"]))
                               $(joinpath(parsed_args["data_dir"], "test.jld2"))`)
