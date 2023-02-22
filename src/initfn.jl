function init(; itasks=[1], utarg_file=nothing, spikerate_file=nothing)
    @assert utarg_file === nothing || spikerate_file === nothing

    # --- initialization --- #
    w0Index, w0Weights, nc0 = genStaticWeights(p.genStaticWeights_args)
    rateX = genRateX(p.genRateX_args)

    itask = 1
    uavg0, ns0, ustd0 = loop_init(itask,
        nothing, nothing, p.stim_off, p.train_time, p.dt,
        p.Nsteps, p.Ncells, p.Ne, nothing, p.LX, p.refrac,
        learn_step, invtau_bale, invtau_bali, nothing, X_bal, nothing,
        nothing, ns, nothing, nsX, inputsE, inputsI, nothing,
        inputsEPrev, inputsIPrev, nothing, nothing, nothing, nothing, nothing,
        u_bale, u_bali, nothing, u_bal, u, nothing, nothing,
        X, nothing, nothing, lastSpike, nothing, nothing, nothing, nothing,
        nothing, v, p.rng, noise, rndX, sig, nothing, w0Index,
        w0Weights, nc0, nothing, nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, uavg, ustd, rateX, cellModel_args)

    wpWeightX, wpWeightIn, wpIndexIn, ncpIn =
        genPlasticWeights(p.genPlasticWeights_args, ns0)

    # get indices of postsynaptic cells for each presynaptic cell
    wpIndexConvert = zeros(Int, p.Ncells, p.Lexc+p.Linh)
    wpIndexOutV = Vector{Int}[Int[] for _ in 1:p.Ncells]
    for postCell = 1:p.Ncells
        for i = 1:ncpIn[postCell]
            preCell = wpIndexIn[i,postCell]
            push!(wpIndexOutV[preCell], postCell)
            wpIndexConvert[postCell,i] = length(wpIndexOutV[preCell])
        end
    end
    ncpOut = [length(wpIndexOutV[preCell]) for preCell = 1:p.Ncells]

    # get weight, index of outgoing connections
    wpIndexOut = zeros(Int, maximum(ncpOut), p.Ncells)
    for preCell = 1:p.Ncells
        wpIndexOut[1:ncpOut[preCell],preCell] = wpIndexOutV[preCell]
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
        utarg = Array{Float64}(undef, Ntime, p.Ncells, length(itasks))
        for itask = 1:length(itasks)
            utarg[:,:,itask] = genUTarget(p.genUTarget_args, uavg0)
        end
    end
    timeSteps = round(Int, (p.stim_off - p.stim_on) / p.dt)
    X_stim = Array{Float64}(undef, timeSteps, p.Ncells, length(itasks))
    for itask = 1:length(itasks)
        X_stim[:,:,itask] = genXStim(p.genXStim_args)
    end

    # --- set up correlation matrix --- #
    pLrec = p.Lexc + p.Linh;

    # L2-penalty
    Pinv_L2 = Diagonal(repeat([p.penlambda], pLrec));
    # row sum penalty
    vec10 = [ones(p.Lexc); zeros(p.Linh)];
    vec01 = [zeros(p.Lexc); ones(p.Linh)];
    Pinv_rowsum = p.penmu*(vec10*vec10' + vec01*vec01');
    # sum of penalties
    Pinv_rec = Pinv_L2 + Pinv_rowsum;
    Pinv_X = Diagonal(repeat([p.penlamFF], p.LX))
    Pinv = zeros(pLrec+p.LX, pLrec+p.LX)
    Pinv[1:pLrec, 1:pLrec] = Pinv_rec
    Pinv[pLrec+1 : pLrec+p.LX, pLrec+1 : pLrec+p.LX] = Pinv_X
    Pinv_norm = p.PType(Symmetric(UpperTriangular(Pinv) \ I));

    #----------- save initialization --------------#
    save(joinpath(data_dir,"w0Index.jld2"), "w0Index", w0Index)
    save(joinpath(data_dir,"w0Weights.jld2"), "w0Weights", w0Weights)
    save(joinpath(data_dir,"nc0.jld2"), "nc0", nc0)
    save(joinpath(data_dir,"X_stim.jld2"), "X_stim", X_stim)
    save(joinpath(data_dir,"utarg.jld2"), "utarg", utarg)
    save(joinpath(data_dir,"wpIndexIn.jld2"), "wpIndexIn", wpIndexIn)
    save(joinpath(data_dir,"wpIndexOut.jld2"), "wpIndexOut", wpIndexOut)
    save(joinpath(data_dir,"wpIndexConvert.jld2"), "wpIndexConvert", wpIndexConvert)
    save(joinpath(data_dir,"wpWeightX.jld2"), "wpWeightX", wpWeightX)
    save(joinpath(data_dir,"wpWeightIn.jld2"), "wpWeightIn", wpWeightIn)
    save(joinpath(data_dir,"ncpIn.jld2"), "ncpIn", ncpIn)
    save(joinpath(data_dir,"ncpOut.jld2"), "ncpOut", ncpOut)
    save(joinpath(data_dir,"P.jld2"), "P", Pinv_norm)
    save(joinpath(data_dir,"rateX.jld2"), "rateX", rateX)

    return (; w0Index, w0Weights, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightX, wpWeightIn,
              nc0, X_stim, utarg, ncpIn, ncpOut, :P => Pinv_norm, rateX)
end
