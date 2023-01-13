function funRollingAvg(startInd,endInd,movavg,cnt,x)
    movavg[startInd:endInd,:] .+= transpose(x)
    cnt[startInd:endInd,:] .+= 1
end

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

@eval function $(Symbol("loop_",kind))(itask, learn_every, stim_on, stim_off,
    train_time, dt, Nsteps, Ncells, L, Ne, Lei, refrac, vre, invtauedecay,
    invtauidecay, invtaudecay_plastic, mu, thresh, tau, maxTimes, times,
    ns, times_ffwd, ns_ffwd, forwardInputsE, forwardInputsI, forwardInputsP,
    forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev, forwardSpike,
    forwardSpikePrev, xedecay, xidecay, xpdecay, synInputBalanced, synInput,
    r, s, bias, wid, example_neurons, lastSpike, bnotrefrac, bspike, plusone,
    minusone, PScale, raug, k, den, e, delta, v, rng, noise, rndFfwd, sig,
    P, Px, w0Index, w0Weights, nc0, stim, xtarg, wpWeightFfwd, wpIndexIn,
    wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ffwdRate)

    @static kind in [:train, :train_test] && p.Lffwd>0 && (ffwdRate /= round(Int, 1000/dt))

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
        @static if p.Lffwd>0
            ffwdSpikePrev .= 0
            s .= 0
        end
    end

    @static if kind in [:test, :train_test]
        times .= 0
        @static p.Lffwd>0 && (times_ffwd .= 0)
    end

    ns .= 0
    @static p.Lffwd>0 && (ns_ffwd .= 0)
    lastSpike .= -100.0
    randn!(rng, v)
    @static if p.K>0
        xedecay .= xidecay .= 0
        forwardInputsEPrev .= forwardInputsIPrev .= 0.0
    end
    xpdecay .= 0
    forwardInputsPPrev .= 0.0
    @static if p.sig>0
        @static if p.noise_model==:voltage 
            sqrtdt = sqrt(dt)
            sqrtinvtau = sqrt.(1 ./ tau)
        elseif p.noise_model==:current
            invsqrtdt = 1/sqrt(dt)
            sqrttau = sqrt.(tau)
        end
    end
    invtau = 1 ./ tau

    # start the actual training
    for ti=1:Nsteps
        t = dt*ti;

        # reset spiking activities from the previous time step
        @static p.K>0 && (forwardInputsE .= forwardInputsI .= 0.0)
        forwardInputsP .= 0.0

        # modify the plastic weights when the stimulus is turned off 
        @static kind in [:train, :train_test] && if t > stim_off && t <= train_time && mod(ti, learn_step) == 0
            wpWeightIn, wpWeightOut = rls(itask, raug, k, den, e, delta, L, Ncells, Lei, r, s, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightFfwd, wpWeightIn, wpWeightOut, plusone, minusone, exactlyzero)
            learn_seq += 1
        end

        @static p.sig>0 && randn!(rng, noise)
        synInputBalanced .= 0.0

        # update network activities
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

        @static p.K>0 && (synInputBalanced .+= xedecay .+ xidecay)
        @static if p.sig>0 && p.noise_model==:current
            synInputBalanced .+= invsqrtdt .* sqrttau .* sig .* noise
        end
        synInput .= synInputBalanced .+ xpdecay

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

        # compute synapse-filtered spike trains
        @static if kind in [:train, :train_test]
            @static if typeof(p.taudecay_plastic)<:Number
                axpby!(invtaudecay_plastic, forwardSpikePrev, plusone-dt*invtaudecay_plastic, r)
            else
                r .+= (-dt.*r .+ forwardSpikePrev) .* invtaudecay_plastic
            end
        end

        # apply external inputs
        if t > stim_on && t < stim_off
            bias .= mu .+ stim[ti-round(Int,stim_on/dt),:,itask]
        else
            bias .= mu
        end

        @static if p.sig>0 && p.noise_model==:voltage
            v .+= sqrtdt .* sqrtinvtau .* sig .* noise
        end

        # not in refractory period
        bnotrefrac .= t .> (lastSpike .+ refrac)
        v .+= bnotrefrac .* dt .* invtau .* (bias .- v .+ synInput)

        # spike occurred
        bspike .= bnotrefrac .& (v .> thresh)
        @static kind in [:train, :train_test] && (forwardSpike .= bspike)
        ns .+= bspike
        v .= ifelse.(bspike, vre, v)
        lastSpike .= ifelse.(bspike, t, lastSpike)
        @static if kind in [:test, :train_test]
            times[ CartesianIndex.(1:Ncells, min.(maxTimes, bspike.*ns) .+ 1) ] .= t
        end

        # accumulate the contribution of spikes to postsynaptic currents
        update_forwardInputs(bspike,
                             w0Index, w0Weights, forwardInputsE, forwardInputsI,
                             wpIndexOut, wpWeightOut, forwardInputsP)

        # external input to trained excitatory neurons
        @static p.Lffwd>0 && if t > stim_off
            @static if kind in [:train, :train_test]
                @static if typeof(p.taudecay_plastic)<:Number
                    axpby!(invtaudecay_plastic, ffwdSpikePrev, plusone-dt*invtaudecay_plastic, s)
                else
                    s .+= (-dt.*s .+ ffwdSpikePrev) .* invtaudecay_plastic
                end
            end

            tidx = ti - round(Int, stim_off/dt)
            rand!(rng, rndFfwd)
            # feed-forward neuron spiked
            bspike_ffwd .= rndFfwd .< @view ffwdRate[tidx,:]
            @static kind in [:train, :train_test] && (ffwdSpike .= bspike_ffwd)
            ns_ffwd .+= bspike_ffwd
            @static if kind in [:test, :train_test]
                times_ffwd[ CartesianIndex.(1:p.Lffwd, min.(maxTimes, bspike_ffwd.*ns_ffwd) .+ 1) ] .= t
            end
            forwardInputsP[2:end] .+= dropdims(sum(view(wpWeightFfwd,:,bspike_ffwd), dims=2), dims=2)
        end

        # save spiking activities produced at the current time step
        @static if p.K>0
            forwardInputsEPrev .= forwardInputsE
            forwardInputsIPrev .= forwardInputsI
        end
        forwardInputsPPrev .= forwardInputsP
        @static if kind in [:train, :train_test]
            forwardSpikePrev .= forwardSpike
            @static p.Lffwd>0 && (ffwdSpikePrev .= ffwdSpike)
        end
    end

    @static if kind in [:test, :train_test]
        xtotal ./= xtotalcnt
        xebal ./= xebalcnt
        xibal ./= xibalcnt
        xplastic ./= xplasticcnt

        return ns, times[:,2:end], ns_ffwd, times_ffwd[:,2:end], xtotal, xebal, xibal, xplastic, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell
    end

end
