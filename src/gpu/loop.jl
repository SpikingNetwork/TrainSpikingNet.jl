function update_forwardInputs(bspike,
                              w0Index, w0Weights, forwardInputsE, forwardInputsI,
                              wpIndexOut, wpWeightOut, forwardInputsP)

    function kernel(bspike,
                    w0Index, w0Weights, forwardInputsE, forwardInputsI,
                    wpIndexOut, wpWeightOut, forwardInputsP)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        @inbounds for i=i0:istride:length(bspike)
            if i<=length(bspike) && bspike[i]
                for j=j0:jstride:max(size(w0Index,1),size(wpIndexOut,1))
                    @static if p.K>0
                        if j<=size(w0Index,1)
                            CUDA.@atomic forwardInputsE[0x1 + w0Index[j,i]] += max(w0Weights[j,i], 0)
                            CUDA.@atomic forwardInputsI[0x1 + w0Index[j,i]] += min(w0Weights[j,i], 0)
                        end
                    end
                    if j<=size(wpIndexOut,1)
                        CUDA.@atomic forwardInputsP[0x1 + wpIndexOut[j,i]] += wpWeightOut[j,i]
                    end
                end
            end
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(bspike,
                                       w0Index, w0Weights, forwardInputsE, forwardInputsI,
                                       wpIndexOut, wpWeightOut, forwardInputsP)
    config = launch_configuration(kernel.fun)
    dims = (length(bspike),
            (@static p.K>0 ? max(size(w0Index,1),size(wpIndexOut,1)) : size(wpIndexOut,1)))
    xthreads = min(32, dims[1])
    ythreads = min(fld(config.threads, xthreads), cld(prod(dims), xthreads))
    xblocks = min(config.blocks, cld(dims[1], xthreads))
    yblocks = min(cld(config.blocks, xblocks), cld(dims[2], ythreads))
    kernel(bspike,
           w0Index, w0Weights, forwardInputsE, forwardInputsI,
           wpIndexOut, wpWeightOut, forwardInputsP;
           threads=(xthreads,ythreads), blocks=(xblocks<<2,yblocks<<2))
end

@eval function $(Symbol("loop_",kind))(learn_every, stim_on, stim_off,
    train_time, dt, Nsteps, Ncells, L, Ne, refrac, vre, invtauedecay,
    invtauidecay, invtaudecay_plastic, mu, thresh, invtau, maxTimes, times,
    ns, forwardInputsE, forwardInputsI, forwardInputsP, forwardInputsEPrev,
    forwardInputsIPrev, forwardInputsPPrev, forwardSpike, forwardSpikePrev,
    xedecay, xidecay, xpdecay, synInputBalanced, synInput, r, bias, wid,
    example_neurons, lastSpike, bnotrefrac, bspike, plusone, minusone, PScale, k,
    den, e, delta, v, rng, noise, sig, P, Px, w0Index, w0Weights, nc0, stim,
    xtarg, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut,
    uavg, utmp)

@static if kind in [:test, :train_test]
    learn_nsteps = round(Int, (train_time - stim_off)/learn_every)
    widInc = round(Int, 2*wid/learn_every - 1)

    vtotal_exccell = CUDA.zeros(eltype(synInput), Nsteps,example_neurons)
    vtotal_inhcell = CUDA.zeros(eltype(synInput), Nsteps,example_neurons)
    vebal_exccell = CUDA.zeros(eltype(synInput), Nsteps,example_neurons)
    vibal_exccell = CUDA.zeros(eltype(synInput), Nsteps,example_neurons)
    vebal_inhcell = CUDA.zeros(eltype(synInput), Nsteps,example_neurons)
    vibal_inhcell = CUDA.zeros(eltype(synInput), Nsteps,example_neurons)
    vplastic_exccell = CUDA.zeros(eltype(synInput), Nsteps,example_neurons)
    vplastic_inhcell = CUDA.zeros(eltype(synInput), Nsteps,example_neurons)

    xtotal = CUDA.zeros(eltype(synInput), learn_nsteps,Ncells)
    xebal = CUDA.zeros(eltype(synInput), learn_nsteps,Ncells)
    xibal = CUDA.zeros(eltype(synInput), learn_nsteps,Ncells)
    xplastic = CUDA.zeros(eltype(synInput), learn_nsteps,Ncells)
    xtotalcnt = CUDA.zeros(Int, learn_nsteps)
    xebalcnt = CUDA.zeros(Int, learn_nsteps)
    xibalcnt = CUDA.zeros(Int, learn_nsteps)
    xplasticcnt = CUDA.zeros(Int, learn_nsteps)
end

@static if kind in [:train, :train_test]
    learn_seq = 1
    r .= 0
    forwardSpikePrev .= 0
end

@static kind in [:test, :train_test] && (times .= 0)

ns .= 0
lastSpike .= -100.0
randn!(rng, v)
xedecay .= xidecay .= xpdecay .= 0
@static if p.K>0
    forwardInputsEPrev .= forwardInputsIPrev .= forwardInputsPPrev .= 0.0
else
    synInputBalanced .= 0.0
end
@static p.K==0 && (sqrtdt = sqrt(dt))

for ti=1:Nsteps
    t = p.dt*ti;

    @static p.K>0 && (forwardInputsE .= forwardInputsI .= 0.0)
    forwardInputsP .= 0.0

    @static kind in [:train, :train_test] && if t > stim_off && t <= train_time && mod(ti, learn_step) == 0
        wpWeightIn, wpWeightOut = rls(k, den, e, delta, L, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone, exactlyzero)
        learn_seq += 1
    end

    @static if p.K>0
        axpby!(invtauedecay, (@view forwardInputsEPrev[2:end]),
               plusone-dt*invtauedecay, xedecay)
        axpby!(invtauidecay, (@view forwardInputsIPrev[2:end]),
               plusone-dt*invtauidecay, xidecay)
    end
    @static if typeof(p.taudecay_plastic)<:Number
        axpby!(invtaudecay_plastic, (@view forwardInputsPPrev[2:end]),
               plusone-dt*invtaudecay_plastic, xpdecay)
    else
        xpdecay .+= (-dt.*xpdecay .+ (@view forwardInputsPPrev[2:end])) .* invtaudecay_plastic
    end

    @static if p.K>0
        synInputBalanced .= xedecay .+ xidecay
        synInput .= synInputBalanced .+ xpdecay
    else
        synInput .= xpdecay
    end

    @static if kind in [:test, :train_test]
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

    @static if kind in [:train, :train_test]
        @static if typeof(p.taudecay_plastic)<:Number
            axpby!(invtaudecay_plastic, forwardSpikePrev, plusone-dt*invtaudecay_plastic, r)
        else
            r .+= (-dt.*r .+ forwardSpikePrev) .* invtaudecay_plastic
        end
    end

    if t > stim_on && t < stim_off
        bias .= mu .+ stim[ti-round(Int,stim_on/p.dt),:]
    else
        bias .= mu
    end

    @static if p.K==0
        randn!(rng, noise)
        v .+= sqrtdt.*sig.*noise
    end

    bnotrefrac .= t .> (lastSpike .+ refrac)
    v .+= bnotrefrac.*dt.*invtau.*(bias .- v .+ synInput)

    bspike .= bnotrefrac .& (v .> thresh)
    @static kind in [:train, :train_test] && (forwardSpike .= bspike)
    ns .+= bspike
    v .= ifelse.(bspike, vre, v)
    lastSpike .= ifelse.(bspike, t, lastSpike)
    @static kind in [:test, :train_test] && (times[ CartesianIndex.(1:Ncells,
                                                     min.(maxTimes, bspike.*ns) .+ 1) ] .= t)

    update_forwardInputs(bspike,
                         w0Index, w0Weights, forwardInputsE, forwardInputsI,
                         wpIndexOut, wpWeightOut, forwardInputsP)

    @static if p.K>0
        forwardInputsEPrev .= forwardInputsE
        forwardInputsIPrev .= forwardInputsI
    end
    forwardInputsPPrev .= forwardInputsP
    @static kind in [:train, :train_test] && (forwardSpikePrev .= forwardSpike)
end

@static if kind in [:test, :train_test]
    xtotal ./= xtotalcnt
    xebal ./= xebalcnt
    xibal ./= xibalcnt
    xplastic ./= xplasticcnt

    return ns, times[:,2:end], xtotal, xebal, xibal, xplastic, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell
end

end
