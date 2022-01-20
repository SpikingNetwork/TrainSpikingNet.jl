using LinearAlgebra, Random, JLD2, Statistics, StatsBase, ArgParse, SymmetricFormats

s = ArgParseSettings()

@add_arg_table! s begin
    "--xtarg_file", "-x"
        help = "full path to the JLD file containing the synaptic targets.  default is sinusoids"
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
end

parsed_args = parse_args(s)

# --- set up variables --- #
include(joinpath(@__DIR__,"struct.jl"))
include(joinpath(parsed_args["data_dir"],"param.jl"))
include(joinpath(@__DIR__,"cpu","variables.jl"))

# --- load code --- #
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

kind=:init
include(joinpath(@__DIR__,"genInitialWeights.jl"))
include(joinpath(@__DIR__,"genPlasticWeights.jl"))
include(joinpath(@__DIR__,"gpu","convertWgtIn2Out.jl"))
include(joinpath(@__DIR__,"genTarget.jl"))
include(joinpath(@__DIR__,"genStim.jl"))
include(joinpath(@__DIR__,"cpu","loop.jl"))
include(joinpath(@__DIR__,"funSample.jl"))

#----------- initialization --------------#
w0Index, w0Weights, nc0 = genInitialWeights(p)

uavg, ns0, ustd = loop_init(nothing, nothing, nothing, p.train_time, dt,
    p.Nsteps, p.Ncells, p.Ne, refrac, vre, invtauedecay, invtauidecay,
    nothing, mu, thresh, invtau, nothing, nothing, ns, forwardInputsE,
    forwardInputsI, nothing, forwardInputsEPrev, forwardInputsIPrev, nothing,
    nothing, nothing, xedecay, xidecay, nothing, nothing, synInput, nothing,
    bias, nothing, nothing, lastSpike, nothing, nothing, nothing, v, rng,
    noise, sig, nothing, nothing, w0Index, w0Weights, nc0, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing, nothing, nothing, uavg, utmp)

wpWeightIn, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut =
    genPlasticWeights(p, w0Index, nc0, ns0)

if isnothing(parsed_args["xtarg_file"])
  xtarg = genTarget(p,uavg,"zero")
else
  xtarg_dict = load(parsed_args["xtarg_file"])
  xtarg = xtarg_dict[first(keys(xtarg_dict))]
end
stim = genStim(p)

#----------- save initialization --------------#
save(joinpath(parsed_args["data_dir"],"p.jld2"), "p", p)
save(joinpath(parsed_args["data_dir"],"w0Index.jld2"), "w0Index", w0Index)
save(joinpath(parsed_args["data_dir"],"w0Weights.jld2"), "w0Weights", w0Weights)
save(joinpath(parsed_args["data_dir"],"nc0.jld2"), "nc0", nc0)
save(joinpath(parsed_args["data_dir"],"stim.jld2"), "stim", stim)
save(joinpath(parsed_args["data_dir"],"xtarg.jld2"), "xtarg", xtarg)
save(joinpath(parsed_args["data_dir"],"wpIndexIn.jld2"), "wpIndexIn", wpIndexIn)
save(joinpath(parsed_args["data_dir"],"wpIndexOut.jld2"), "wpIndexOut", wpIndexOut)
save(joinpath(parsed_args["data_dir"],"wpIndexConvert.jld2"), "wpIndexConvert", wpIndexConvert)
save(joinpath(parsed_args["data_dir"],"wpWeightIn.jld2"), "wpWeightIn", wpWeightIn)
save(joinpath(parsed_args["data_dir"],"ncpIn.jld2"), "ncpIn", ncpIn)
save(joinpath(parsed_args["data_dir"],"ncpOut.jld2"), "ncpOut", ncpOut)
