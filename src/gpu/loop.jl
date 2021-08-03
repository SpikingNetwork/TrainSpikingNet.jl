function kernelEI(ispike, w0Index, w0Weights, forwardInputsE, forwardInputsI)
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    istride = blockDim().x * gridDim().x
    jstride = blockDim().y * gridDim().y

    @inbounds for i=i0:istride:size(w0Index,1), j=j0:jstride:length(ispike)
        @atomic forwardInputsE[0x1 + w0Index[i,ispike[j]]] += max(w0Weights[i,ispike[j]], 0)
        @atomic forwardInputsI[0x1 + w0Index[i,ispike[j]]] += min(w0Weights[i,ispike[j]], 0)
    end
    return nothing
end

function kernelP(ispike, wpIndexOut, wpWeightOut, forwardInputsP)
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    istride = blockDim().x * gridDim().x
    jstride = blockDim().y * gridDim().y

    @inbounds for i=i0:istride:size(wpIndexOut,1), j=j0:jstride:length(ispike)
        @atomic forwardInputsP[0x1 + wpIndexOut[i,ispike[j]]] += wpWeightOut[i,ispike[j]]
    end
    return nothing
end

function configurator(config, size_weights)
    xthreads = min(32, size_weights[1])
    ythreads = min(cld(config.threads, xthreads), cld(prod(size_weights), xthreads))
    xblocks = cld(size_weights[1], xthreads)
    yblocks = cld(size_weights[2], ythreads)

    return (threads=(xthreads, ythreads), blocks=(xblocks, yblocks))
end

cukernelEI = cufunction(kernelEI, Tuple{CuDeviceArray{UInt64,1,AS.Global}, CuDeviceArray{p.IntPrecision,2,AS.Global}, CuDeviceArray{p.FloatPrecision,2,AS.Global}, CuDeviceArray{p.FloatPrecision,1,AS.Global}, CuDeviceArray{p.FloatPrecision,1,AS.Global}})

cukernelP = cufunction(kernelP, Tuple{CuDeviceArray{UInt64,1,AS.Global}, CuDeviceArray{p.IntPrecision,2,AS.Global}, CuDeviceArray{p.FloatPrecision,2,AS.Global}, CuDeviceArray{p.FloatPrecision,1,AS.Global}})

@eval function $(Symbol("loop_",kind))(learn_every, stim_on, stim_off,
    train_time, dt, Nsteps, Ncells, L, Ne, refrac, vre, invtauedecay,
    invtauidecay, invtaudecay_plastic, mu, thresh, invtau, ns, forwardInputsE,
    forwardInputsI, forwardInputsP, forwardInputsEPrev, forwardInputsIPrev,
    forwardInputsPPrev, forwardSpike, forwardSpikePrev, xedecay, xidecay,
    xpdecay, synInputBalanced, synInput, r, bias, wid, example_neurons,
    lastSpike, bnotrefrac, bspike, plusone, minusone, k, den, e, v, P,
    Px, w0Index, w0Weights, nc0, stim, xtarg, wpIndexIn, wpIndexOut,
    wpIndexConvert, wpWeightIn, wpWeightOut, ncpOut, uavg, utmp)

@static if kind == :test
    learn_nsteps = Int((train_time - stim_off)/learn_every)
    widInc = Int(2*wid/learn_every - 1)

    vtotal_exccell = CUDA.zeros(Nsteps,example_neurons)
    vtotal_inhcell = CUDA.zeros(Nsteps,example_neurons)
    vebal_exccell = CUDA.zeros(Nsteps,example_neurons)
    vibal_exccell = CUDA.zeros(Nsteps,example_neurons)
    vebal_inhcell = CUDA.zeros(Nsteps,example_neurons)
    vibal_inhcell = CUDA.zeros(Nsteps,example_neurons)
    vplastic_exccell = CUDA.zeros(Nsteps,example_neurons)
    vplastic_inhcell = CUDA.zeros(Nsteps,example_neurons)

    xtotal = CUDA.zeros(learn_nsteps,Ncells)
    xebal = CUDA.zeros(learn_nsteps,Ncells)
    xibal = CUDA.zeros(learn_nsteps,Ncells)
    xplastic = CUDA.zeros(learn_nsteps,Ncells)
    xtotalcnt = CUDA.zeros(learn_nsteps)
    xebalcnt = CUDA.zeros(learn_nsteps)
    xibalcnt = CUDA.zeros(learn_nsteps)
    xplasticcnt = CUDA.zeros(learn_nsteps)
end

@static if kind == :train
    learn_seq = 1
    r .= 0
end

ns .= 0
lastSpike .= -100.0
v .= CuArray(rand(p.Ncells))
xedecay .= xidecay .= xpdecay .= 0
forwardInputsEPrev .= forwardInputsIPrev .= forwardInputsPPrev .= 0.0

for ti=1:Nsteps
    t = dt*ti;

    forwardInputsE .= 0.0;
    forwardInputsI .= 0.0;
    forwardInputsP .= 0.0;

    @static kind==:train && if t > stim_off && t <= train_time && mod(t, learn_every) == 0
        wpWeightIn, wpWeightOut = rls(k, den, e, L, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone)
        learn_seq += 1
    end

    xedecay .+= (-dt.*xedecay .+ forwardInputsEPrev[2:end]).*invtauedecay
    xidecay .+= (-dt.*xidecay .+ forwardInputsIPrev[2:end]).*invtauidecay
    xpdecay .+= (-dt.*xpdecay .+ forwardInputsPPrev[2:end]).*invtaudecay_plastic
    synInputBalanced .= xedecay .+ xidecay
    synInput .= synInputBalanced .+ xpdecay

    @static if kind == :test
        # saved for visualization
        vtotal_exccell[ti,1:example_neurons] .= synInput[1:example_neurons]
        vebal_exccell[ti,1:example_neurons] .= xedecay[1:example_neurons]
        vibal_exccell[ti,1:example_neurons] .= xidecay[1:example_neurons]
        vplastic_exccell[ti,1:example_neurons] .= xpdecay[1:example_neurons]
        vtotal_inhcell[ti,1:example_neurons] .= synInput[1:example_neurons]
        vebal_inhcell[ti,1:example_neurons] .= xedecay[end-example_neurons+1:end]
        vibal_inhcell[ti,1:example_neurons] .= xidecay[end-example_neurons+1:end]
        vplastic_inhcell[ti,1:example_neurons] .= xpdecay[end-example_neurons+1:end]

        # save rolling average for analysis
        if t > stim_off && t <= train_time && mod(t,1.0) == 0
            startInd = floor(Int, (t - stim_off - wid)/learn_every + 1)
            endInd = min(startInd + widInc, learn_nsteps)
            startInd = max(startInd, 1)
            funRollingAvg(startInd, endInd, xtotal, xtotalcnt, synInput)
            funRollingAvg(startInd, endInd, xebal, xebalcnt, xedecay)
            funRollingAvg(startInd, endInd, xibal, xibalcnt, xidecay)
            funRollingAvg(startInd, endInd, xplastic, xplasticcnt, xpdecay)
        end
    end

    @static if kind == :train
        r .+= (-dt.*r .+ forwardSpikePrev).*invtaudecay_plastic
    end

    if t > stim_on && t < stim_off
        bias .= mu .+ stim[ti-round(Int,stim_on/dt),:]
    else
        bias .= mu
    end

    bnotrefrac .= t .> (lastSpike .+ refrac)
    v .+= bnotrefrac.*dt.*(invtau.*(bias .- v .+ synInput))

    bspike .= bnotrefrac .& (v .> thresh)
    @static kind==:train && (forwardSpike .= bspike)
    ns .+= bspike
    v .= ifelse.(bspike, vre, v)
    lastSpike .= ifelse.(bspike, t, lastSpike)

    ispike = findall(bspike)
    if length(ispike)>0
        configEI = configurator(CUDA.launch_configuration(cukernelEI.fun),
                                (size(w0Weights,1),length(ispike)))
        @cuda name="update_forwardInputsEI" threads=configEI.threads blocks=configEI.blocks kernelEI(ispike, w0Index, w0Weights, forwardInputsE, forwardInputsI)

        configP = configurator(CUDA.launch_configuration(cukernelP.fun),
                               (size(wpWeightOut,1),length(ispike)))
        @cuda name="update_forwardInputsP" threads=configP.threads blocks=configP.blocks kernelP(ispike, wpIndexOut, wpWeightOut, forwardInputsP)
    end

    forwardInputsEPrev .= forwardInputsE
    forwardInputsIPrev .= forwardInputsI
    forwardInputsPPrev .= forwardInputsP
    @static kind==:train && (forwardSpikePrev .= forwardSpike)
end

@static if kind == :test
    xtotal = xtotal ./ xtotalcnt
    xebal = xebal ./ xebalcnt
    xibal = xibal ./ xibalcnt
    xplastic = xplastic ./ xplasticcnt

    return xtotal, xebal, xibal, xplastic, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell
end

end
