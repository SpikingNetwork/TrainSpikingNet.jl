using Pkg;  Pkg.activate(dirname(@__DIR__), io=devnull)

using LinearAlgebra, Random, JLD2, Statistics, StatsBase, ArgParse, SymmetricFormats, UnPack

# --- define command line arguments --- #
function ArgParse.parse_item(::Type{Vector{Int}}, x::AbstractString)
    return eval(Meta.parse(x))
end

aps = ArgParseSettings()

@add_arg_table! aps begin
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
    "--itasks", "-t"
        help = "a vector specifying which tasks to learn"
        arg_type = Vector{Int}
        default = [1]
        range_tester = x->all(x.>0)
end

add_arg_group!(aps, "mutually exclusive arguments.  if neither is specified, sinusoids\nwill be generated for synpatic inputs", exclusive = true);

@add_arg_table! aps begin
    "--xtarg_file", "-x"
        help = "full path to the JLD file containing the synaptic current targets"
    "--spikerate_file", "-s"
        help = "full path to the JLD file containing the spike rates"
end

parsed_args = parse_args(aps)

# --- set up variables --- #
module Param
    using LinearAlgebra, SymmetricFormats, Random, JLD2
    function set_data_dir(path)
        global data_dir = path
    end
    save_params() = save(joinpath(data_dir, "param.jld2"), "param", (;
            PPrecision, PScale, FloatPrecision, IntPrecision, PType,
            seed, rng_func,
            example_neurons, wid, maxrate,
            penlambda, penlamFF, penmu,
            learn_every,
            train_duration, stim_on, stim_off, train_time,
            dt, Nsteps,
            Ncells, Ne, Ni,
            taue, taui,
            K, L, Lffwd, Lexc, Linh,
            mu,
            vre, threshe, threshi, refrac,
            tauedecay, tauidecay, taudecay_plastic,
            noise_model, sig,
            correlation_var,
            genStim_file, genStim_args,
            genTarget_file, genTarget_args,
            genFfwdRate_file, genFfwdRate_args,
            genStaticWeights_file, genStaticWeights_args,
            genPlasticWeights_file, genPlasticWeights_args,
            choose_task_func,
            ))
end
Param.set_data_dir(parsed_args["data_dir"])
Param.include(joinpath(parsed_args["data_dir"], "param.jl"))
Param.save_params()
include(joinpath(@__DIR__,"cpu","variables.jl"))

if Param.Ncells == typemax(Param.IntPrecision)
  @warn "IntPrecision is too small for GPU (but fine for CPU)"
elseif Param.Ncells > typemax(Param.IntPrecision)
  @error "IntPrecision is too small"
end

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
include(Param.genStaticWeights_file)
include(Param.genPlasticWeights_file)
include(Param.genFfwdRate_file)
include(Param.genTarget_file)
include(Param.genStim_file)
include(joinpath("cpu","loop.jl"))
include("rate2synInput.jl")

# --- initialization --- #
w0Index, w0Weights, nc0 = genStaticWeights(Param.genStaticWeights_args)
ffwdRate = genFfwdRate(Param.genFfwdRate_args)

itask = 1
uavg, ns0, ustd = loop_init(itask,
    nothing, nothing, Param.stim_off, Param.train_time, Param.dt,
    Param.Nsteps, Param.Ncells, Param.Ne, nothing, Param.Lffwd, Param.refrac,
    vre, invtauedecay, invtauidecay, nothing, mu, thresh, tau, nothing,
    nothing, ns, nothing, ns_ffwd, forwardInputsE, forwardInputsI, nothing,
    forwardInputsEPrev, forwardInputsIPrev, nothing, nothing, nothing,
    xedecay, xidecay, nothing, synInputBalanced, synInput, nothing, nothing,
    bias, nothing, nothing, lastSpike, nothing, nothing, nothing, nothing,
    nothing, v, Param.rng, noise, rndFfwd, sig, nothing, nothing, w0Index,
    w0Weights, nc0, nothing, nothing, nothing, nothing, nothing, nothing,
    nothing, nothing, nothing, nothing, uavg, utmp, ffwdRate)

wpWeightFfwd, wpWeightIn, wpIndexIn, ncpIn =
    genPlasticWeights(Param.genPlasticWeights_args, ns0)

# get indices of postsynaptic cells for each presynaptic cell
wpIndexConvert = zeros(Int, Param.Ncells, Param.Lexc+Param.Linh)
wpIndexOutD = Dict{Int,Array{Int,1}}()
ncpOut = Array{Int}(undef, Param.Ncells)
for i = 1:Param.Ncells
    wpIndexOutD[i] = []
end
for postCell = 1:Param.Ncells
    for i = 1:ncpIn[postCell]
        preCell = wpIndexIn[postCell,i]
        push!(wpIndexOutD[preCell], postCell)
        wpIndexConvert[postCell,i] = length(wpIndexOutD[preCell])
    end
end
for preCell = 1:Param.Ncells
    ncpOut[preCell] = length(wpIndexOutD[preCell])
end

# get weight, index of outgoing connections
wpIndexOut = zeros(Int, maximum(ncpOut), Param.Ncells)
for preCell = 1:Param.Ncells
    wpIndexOut[1:ncpOut[preCell],preCell] = wpIndexOutD[preCell]
end

Ntime = floor(Int, (Param.train_time - Param.stim_off) / Param.learn_every)
if parsed_args["xtarg_file"] !== nothing
    xtarg_dict = load(parsed_args["xtarg_file"])
    xtarg = xtarg_dict[first(keys(xtarg_dict))]
    if size(xtarg,1) != Ntime
        error(parsed_args["xtarg_file"],
              " should have (train_time-stim_off)/learn_every = ",
              Ntime, " rows")
    end
    ndims(xtarg)==2 && (xtarg = xtarg[:,:,[CartesianIndex()]])
    if any(parsed_args["itasks"] .> size(xtarg,3))
        error("an element of --itasks exceeds the size of the third dimension of ",
              parsed_args["xtarg_file"])
    end
    xtarg = xtarg[:,:,parsed_args["itasks"]]
elseif parsed_args["spikerate_file"] !== nothing
    xtarg = rate2synInput(Param.train_time, Param.stim_off, Param.learn_every,
                          Param.taue, Param.threshe, Param.vre,
                          Param.K==0 ? sig : (ustd / sqrt(Param.taue_bal * 1.3))) # factor 1.3 was calibrated manually
else
    xtarg = Array{Float64}(undef, Ntime, Param.Ncells, length(parsed_args["itasks"]))
    for itask = 1:length(parsed_args["itasks"])
        xtarg[:,:,itask] = genTarget(Param.genTarget_args, uavg)
    end
end
timeSteps = round(Int, (Param.stim_off - Param.stim_on) / Param.dt)
stim = Array{Float64}(undef, timeSteps, Param.Ncells, length(parsed_args["itasks"]))
for itask = 1:length(parsed_args["itasks"])
    stim[:,:,itask] = genStim(Param.genStim_args)
end

# --- set up correlation matrix --- #
pLrec = Param.Lexc + Param.Linh;

# L2-penalty
Pinv_L2 = Diagonal(repeat([Param.penlambda], pLrec));
# row sum penalty
vec10 = [ones(Param.Lexc); zeros(Param.Linh)];
vec01 = [zeros(Param.Lexc); ones(Param.Linh)];
Pinv_rowsum = Param.penmu*(vec10*vec10' + vec01*vec01');
# sum of penalties
Pinv_rec = Pinv_L2 + Pinv_rowsum;
Pinv_ffwd = Diagonal(repeat([Param.penlamFF], Param.Lffwd))
Pinv = zeros(pLrec+Param.Lffwd, pLrec+Param.Lffwd)
Pinv[1:pLrec, 1:pLrec] = Pinv_rec
Pinv[pLrec+1 : pLrec+Param.Lffwd, pLrec+1 : pLrec+Param.Lffwd] = Pinv_ffwd
Pinv_norm = Param.PType(Symmetric(UpperTriangular(Pinv) \ I));

#----------- save initialization --------------#
save(joinpath(parsed_args["data_dir"],"w0Index.jld2"), "w0Index", w0Index)
save(joinpath(parsed_args["data_dir"],"w0Weights.jld2"), "w0Weights", w0Weights)
save(joinpath(parsed_args["data_dir"],"nc0.jld2"), "nc0", nc0)
save(joinpath(parsed_args["data_dir"],"stim.jld2"), "stim", stim)
save(joinpath(parsed_args["data_dir"],"xtarg.jld2"), "xtarg", xtarg)
save(joinpath(parsed_args["data_dir"],"wpIndexIn.jld2"), "wpIndexIn", wpIndexIn)
save(joinpath(parsed_args["data_dir"],"wpIndexOut.jld2"), "wpIndexOut", wpIndexOut)
save(joinpath(parsed_args["data_dir"],"wpIndexConvert.jld2"), "wpIndexConvert", wpIndexConvert)
save(joinpath(parsed_args["data_dir"],"wpWeightFfwd.jld2"), "wpWeightFfwd", wpWeightFfwd)
save(joinpath(parsed_args["data_dir"],"wpWeightIn.jld2"), "wpWeightIn", wpWeightIn)
save(joinpath(parsed_args["data_dir"],"ncpIn.jld2"), "ncpIn", ncpIn)
save(joinpath(parsed_args["data_dir"],"ncpOut.jld2"), "ncpOut", ncpOut)
save(joinpath(parsed_args["data_dir"],"P.jld2"), "P", Pinv_norm)
save(joinpath(parsed_args["data_dir"],"ffwdRate.jld2"), "ffwdRate", ffwdRate)
