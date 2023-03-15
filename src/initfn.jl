function init(; itasks=[1], utarg_file=nothing, spikerate_file=nothing)
    @assert utarg_file === nothing || spikerate_file === nothing

    # --- initialization --- #
    w0Index, w0Weights = genStaticWeights(p.genStaticWeights_args)
    rateX = genRateX(p.genRateX_args)

    itask = 1
    uavg0, ns0, ustd0 = loop_init(itask,
        nothing, nothing, p.stim_off, p.train_time, p.dt, p.Nsteps,
        u0_skip_steps, p.u0_ncells, p.Ncells, p.Ne, p.LX, p.refrac,
        learn_step, invtau_bale, invtau_bali, nothing, X_bal, nothing,
        nothing, ns, nothing, nsX, inputsE, inputsI, nothing,
        inputsEPrev, inputsIPrev, nothing, nothing, nothing, nothing, nothing,
        u_bale, u_bali, nothing, u_bal, u, nothing, nothing,
        X, nothing, nothing, lastSpike, nothing, nothing, nothing, nothing,
        nothing, nothing, v, p.rng, noise, rndX, sig, nothing, w0Index,
        w0Weights, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, uavg, ustd, rateX, cellModel_args,
        TCurrent, TCharge, TTime)

    wpWeightX, wpWeightIn, wpIndexIn =
        genPlasticWeights(p.genPlasticWeights_args, ns0)

    # get indices of postsynaptic cells for each presynaptic cell
    ncpInMax = maximum([length(x) for x in wpIndexIn])
    wpIndexOut = Vector{Int}[Int[] for _ in 1:p.Ncells]
    wpIndexConvert = Vector{Vector{Int}}(undef, p.Ncells)
    for postCell = 1:p.Ncells
        wpIndexConvert[postCell] = Vector{Int}(undef, length(wpIndexIn[postCell]))
        for i in eachindex(wpIndexIn[postCell])
            preCell = wpIndexIn[postCell][i]
            push!(wpIndexOut[preCell], postCell)
            wpIndexConvert[postCell][i] = length(wpIndexOut[preCell])
        end
    end

    # load or calculate target synaptic currents
    Ntime = floor(Int, (p.train_time - p.stim_off) / p.learn_every)
    if utarg_file !== nothing
        utarg_dict = load(utarg_file)
        utarg = utarg_dict[first(keys(utarg_dict))]
        if size(utarg,1) != Ntime
            error(utarg_file,
                  " should have (train_time-stim_off)/learn_every = ",
                  Ntime, " rows")
        end
        ndims(utarg)==2 && (utarg = utarg[:,:,[CartesianIndex()]])
        if any(itasks .> size(utarg,3))
            error("an element of --itasks exceeds the size of the third dimension of ",
                  utarg_file)
        end
        utarg = utarg[:,:,itasks]
    elseif spikerate_file !== nothing
        utarg = rate2utarg(itasks, spikerate_file,
                           p.train_time, p.stim_off, p.learn_every,
                           p.tau_meme, p.threshe, p.vre,
                           p.K==0 ? p.sig : (ustd0 / sqrt(p.tau_bale * 1.3))) # factor 1.3 was calibrated manually
    else
        utarg = Array{eltype(p.g)}(undef, Ntime, p.Ncells, length(itasks))
        for itask = 1:length(itasks)
            utarg[:,:,itask] = genUTarget(p.genUTarget_args, uavg0)
        end
    end
    timeSteps = round(Int, (p.stim_off - p.stim_on) / p.dt)
    X_stim = Array{eltype(p.g)}(undef, timeSteps, p.Ncells, length(itasks))
    for itask = 1:length(itasks)
        X_stim[:,:,itask] = genXStim(p.genXStim_args)
    end

    # --- set up correlation matrix --- #
    Pinv_X = Diagonal(repeat([p.penlamFF], p.LX))
    P = Array{p.PType}(undef, p.Ncells)
    charge0 = TCharge(0)
    Threads.@threads for i = 1:p.Ncells
        ncpIn = length(wpIndexIn[i])
        Pinv_L2 = Diagonal(repeat([p.penlambda], ncpIn));
        vec10 = wpWeightIn[i] .> charge0
        vec01 = wpWeightIn[i] .< charge0
        Pinv_rowsum = p.penmu*(vec10*vec10' + vec01*vec01')
        Pinv = zeros(p.LX+ncpIn, p.LX+ncpIn)
        Pinv[1:p.LX, 1:p.LX] = Pinv_X
        Pinv[p.LX+1:end, p.LX+1:end] = Pinv_L2 + Pinv_rowsum
        P[i] = p.PType(Symmetric(UpperTriangular(Pinv) \ I))
    end

    #----------- save initialization --------------#
    save(joinpath(data_dir,"w0Index.jld2"), "w0Index", w0Index)
    save(joinpath(data_dir,"w0Weights.jld2"), "w0Weights", w0Weights)
    save(joinpath(data_dir,"X_stim.jld2"), "X_stim", X_stim)
    save(joinpath(data_dir,"utarg.jld2"), "utarg", utarg)
    save(joinpath(data_dir,"wpIndexIn.jld2"), "wpIndexIn", wpIndexIn)
    save(joinpath(data_dir,"wpIndexOut.jld2"), "wpIndexOut", wpIndexOut)
    save(joinpath(data_dir,"wpIndexConvert.jld2"), "wpIndexConvert", wpIndexConvert)
    save(joinpath(data_dir,"wpWeightX.jld2"), "wpWeightX", wpWeightX)
    save(joinpath(data_dir,"wpWeightIn.jld2"), "wpWeightIn", wpWeightIn)
    save(joinpath(data_dir,"P.jld2"), "P", P)
    save(joinpath(data_dir,"rateX.jld2"), "rateX", rateX)

    return (; w0Index, w0Weights, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightX, wpWeightIn,
              X_stim, utarg, P, rateX)
end
