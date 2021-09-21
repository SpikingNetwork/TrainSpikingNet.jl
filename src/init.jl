using LinearAlgebra, Random, JLD, Statistics, StatsBase

data_dir = ARGS[1]
xtarg_file = length(ARGS)>1 ? ARGS[2] : nothing

Random.seed!(1)

# --- set up variables --- #
include(joinpath(@__DIR__,"struct.jl"))
include(joinpath(data_dir,"param.jl"))
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
    bias, nothing, nothing, lastSpike, nothing, nothing, v, rng, noise, sig,
    nothing, nothing, w0Index, w0Weights, nc0, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, nothing, nothing, uavg, utmp)

wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut =
    genPlasticWeights(p, w0Index, nc0, ns0)

if isnothing(xtarg_file)
  xtarg = genTarget(p,uavg,"zero")
else
  xtarg_dict = load(xtarg_file)
  xtarg = xtarg_dict[first(keys(xtarg_dict))]
end
stim = genStim(p)

#----------- save initialization --------------#
save(joinpath(data_dir,"w0Index.jld"), "w0Index", w0Index)
save(joinpath(data_dir,"w0Weights.jld"), "w0Weights", w0Weights)
save(joinpath(data_dir,"nc0.jld"), "nc0", nc0)
save(joinpath(data_dir,"stim.jld"), "stim", stim)
save(joinpath(data_dir,"xtarg.jld"), "xtarg", xtarg)
save(joinpath(data_dir,"wpIndexIn.jld"), "wpIndexIn", wpIndexIn)
save(joinpath(data_dir,"wpIndexOut.jld"), "wpIndexOut", wpIndexOut)
save(joinpath(data_dir,"wpIndexConvert.jld"), "wpIndexConvert", wpIndexConvert)
save(joinpath(data_dir,"wpWeightIn.jld"), "wpWeightIn", wpWeightIn)
save(joinpath(data_dir,"wpWeightOut.jld"), "wpWeightOut", wpWeightOut)
save(joinpath(data_dir,"ncpIn.jld"), "ncpIn", ncpIn)
save(joinpath(data_dir,"ncpOut.jld"), "ncpOut", ncpOut)
