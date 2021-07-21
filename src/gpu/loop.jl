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
    xpdecay, synInputBalanced, r, bias, example_neurons, lastSpike,
    bnotrefrac, bspike, plusone, minusone, k, den, e, v, P, Px, w0Index,
    w0Weights, nc0, stim, xtarg, wpIndexIn, wpIndexOut, wpIndexConvert,
    wpWeightIn, wpWeightOut, ncpOut, uavg, utmp)

@static kind == :train && (learn_seq = 1)

for ti=1:Nsteps
    t = dt*ti;

    forwardInputsE .= 0.0;
    forwardInputsI .= 0.0;
    forwardInputsP .= 0.0;

    if t > stim_off && t <= train_time && mod(t, learn_every) == 0
        wpWeightIn, wpWeightOut, learn_seq = rls(k, den, e, L, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone)
    end

    xedecay .+= (-dt.*xedecay .+ forwardInputsEPrev[2:end]).*invtauedecay
    xidecay .+= (-dt.*xidecay .+ forwardInputsIPrev[2:end]).*invtauidecay
    xpdecay .+= (-dt.*xpdecay .+ forwardInputsPPrev[2:end]).*invtaudecay_plastic
    synInputBalanced .= xedecay .+ xidecay

    r .+= (-dt.*r .+ forwardSpikePrev).*invtaudecay_plastic

    if t > stim_on && t < stim_off
        bias .= mu .+ stim[ti-round(Int,stim_on/dt),:]
    else
        bias .= mu
    end

    bnotrefrac .= t .> (lastSpike .+ refrac)
    v .+= bnotrefrac.*dt.*(invtau.*(bias .- v .+ synInputBalanced .+ xpdecay))

    bspike .= bnotrefrac .& (v .> thresh)
    forwardSpike .= bspike
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

    forwardInputsEPrev = copy(forwardInputsE)
    forwardInputsIPrev = copy(forwardInputsI)
    forwardInputsPPrev = copy(forwardInputsP)
    forwardSpikePrev = copy(forwardSpike)
end

end
