function update_inputs(bspike,
                       w0Index, w0Weights, inputsE, inputsI,
                       wpIndexOut, wpWeightOut, inputsP)

    function kernel(bspike,
                    w0Index, w0Weights, inputsE, inputsI,
                    wpIndexOut, wpWeightOut, inputsP)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        @inbounds for i=i0:istride:length(bspike)
            if i<=length(bspike) && bspike[i]
                for j=j0:jstride:max(size(w0Index,1),size(wpIndexOut,1))
                    @static if p.K>0
                        if j<=size(w0Index,1)
                            CUDA.@atomic inputsE[0x1 + w0Index[j,i]] += max(w0Weights[j,i], 0)
                            CUDA.@atomic inputsI[0x1 + w0Index[j,i]] += min(w0Weights[j,i], 0)
                        end
                    end
                    if j<=size(wpIndexOut,1)
                        CUDA.@atomic inputsP[0x1 + wpIndexOut[j,i]] += wpWeightOut[j,i]
                    end
                end
            end
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(bspike,
                                       w0Index, w0Weights, inputsE, inputsI,
                                       wpIndexOut, wpWeightOut, inputsP)
    config = launch_configuration(kernel.fun)
    dims = (length(bspike),
            (@static p.K>0 ? max(size(w0Index,1),size(wpIndexOut,1)) : size(wpIndexOut,1)))
    xthreads = min(32, dims[1])
    ythreads = min(fld(config.threads, xthreads), cld(prod(dims), xthreads))
    xblocks = min(config.blocks, cld(dims[1], xthreads))
    yblocks = min(cld(config.blocks, xblocks), cld(dims[2], ythreads))
    kernel(bspike,
           w0Index, w0Weights, inputsE, inputsI,
           wpIndexOut, wpWeightOut, inputsP;
           threads=(xthreads,ythreads), blocks=(xblocks<<2,yblocks<<2))
end

@eval function $(Symbol("loop_",kind))(itask,
    learn_every, stim_on, stim_off, train_time, dt, Nsteps, Ncells, Ne, Lei,
    LX, refrac, learn_step, invtau_bale, invtau_bali, invtau_plas, X_bal,
    maxTimes, times, ns, timesX, nsX, inputsE, inputsI,
    inputsP, inputsEPrev, inputsIPrev, inputsPPrev, spikes, spikesPrev,
    spikesX, spikesXPrev, u_bale, u_bali, uX_plas, u_bal,
    u, r, rX, X, wid, example_neurons, lastSpike, bnotrefrac,
    bspike, plusone, minusone, PScale, raug, k, den, e, delta, v, rng, noise,
    rndX, sig, P, w0Index, w0Weights, nc0, X_stim, utarg, wpWeightX,
    wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, rateX,
    cellModel_args)

    @static (kind in [:train, :train_test] && p.LX>0) && (rateX /= round(Int, 1000/dt))

    @static if kind in [:test, :train_test]
        learn_nsteps = round(Int, (train_time - stim_off)/learn_every)
        widInc = round(Int, 2*wid/learn_every - 1)

        u_exccell = CUDA.zeros(eltype(u), Nsteps,example_neurons)
        u_inhcell = CUDA.zeros(eltype(u), Nsteps,example_neurons)
        u_bale_exccell = CUDA.zeros(eltype(u), Nsteps,example_neurons)
        u_bali_exccell = CUDA.zeros(eltype(u), Nsteps,example_neurons)
        u_bale_inhcell = CUDA.zeros(eltype(u), Nsteps,example_neurons)
        u_bali_inhcell = CUDA.zeros(eltype(u), Nsteps,example_neurons)
        u_plas_exccell = CUDA.zeros(eltype(u), Nsteps,example_neurons)
        u_plas_inhcell = CUDA.zeros(eltype(u), Nsteps,example_neurons)

        u_rollave = CUDA.zeros(eltype(u), learn_nsteps,Ncells)
        u_bale_rollave = CUDA.zeros(eltype(u), learn_nsteps,Ncells)
        u_bali_rollave = CUDA.zeros(eltype(u), learn_nsteps,Ncells)
        u_plas_rollave = CUDA.zeros(eltype(u), learn_nsteps,Ncells)
        u_rollave_cnt = CUDA.zeros(Int, learn_nsteps)
    end

    @static if kind in [:train, :train_test]
        learn_seq = 1
        r .= 0
        spikesPrev .= 0
        @static if p.LX>0
            spikesXPrev .= 0
            rX .= 0
        end
    end

    @static if kind in [:test, :train_test]
        times .= 0
        @static p.LX>0 && (timesX .= 0)
    end

    ns .= 0
    @static p.LX>0 && (nsX .= 0)
    lastSpike .= -100.0
    randn!(rng, v)
    @static if p.K>0
        u_bale .= u_bali .= 0
        inputsEPrev .= inputsIPrev .= 0.0
    end
    uX_plas .= 0
    inputsPPrev .= 0.0

    # start the actual training
    for ti=1:Nsteps
        t = dt*ti;

        # reset spiking activities from the previous time step
        @static p.K>0 && (inputsE .= inputsI .= 0.0)
        inputsP .= 0.0

        # modify the plastic weights when the stimulus is turned off 
        @static if kind in [:train, :train_test]
            if t > stim_off && t <= train_time && mod(ti, learn_step) == 0
                wpWeightIn, wpWeightOut = rls(itask,
                        raug, k, den, e, delta, Ncells, Lei, r, rX,
                        P, u_bal, utarg, learn_seq, wpIndexIn,
                        wpIndexConvert, wpWeightX, wpWeightIn, wpWeightOut,
                        plusone, minusone, exactlyzero, PScale)
                learn_seq += 1
            end
        end

        @static p.sig>0 && randn!(rng, noise)
        u_bal .= 0.0

        # update network activities
        @static if p.K>0
            axpby!(invtau_bale, (@view inputsEPrev[2:end]), plusone-dt*invtau_bale, u_bale)
            axpby!(invtau_bali, (@view inputsIPrev[2:end]), plusone-dt*invtau_bali, u_bali)
        end
        @static if typeof(p.tau_plas)<:Number
            axpby!(invtau_plas, (@view inputsPPrev[2:end]), plusone-dt*invtau_plas, uX_plas)
        else
            uX_plas .+= (-dt.*uX_plas .+ (@view inputsPPrev[2:end])) .* invtau_plas
        end

        @static p.K>0 && (u_bal .+= u_bale .+ u_bali)
        @static if p.sig>0 && p.noise_model==:current
            u_bal .+= sig .* noise
        end
        u .= u_bal .+ uX_plas

        @static if kind in [:test, :train_test]
            # saved for visualization
            u_exccell[ti,1:example_neurons] .= u[1:example_neurons]
            u_bale_exccell[ti,1:example_neurons] .= u_bale[1:example_neurons]
            u_bali_exccell[ti,1:example_neurons] .= u_bali[1:example_neurons]
            u_plas_exccell[ti,1:example_neurons] .= uX_plas[1:example_neurons]
            u_inhcell[ti,1:example_neurons] .= u[1:example_neurons]
            u_bale_inhcell[ti,1:example_neurons] .= u_bale[end-example_neurons+1:end]
            u_bali_inhcell[ti,1:example_neurons] .= u_bali[end-example_neurons+1:end]
            u_plas_inhcell[ti,1:example_neurons] .= uX_plas[end-example_neurons+1:end]

            # save rolling average for analysis
            if t > stim_off && t <= train_time && mod(t,1.0) == 0
                startInd = floor(Int, (t - stim_off - wid)/learn_every + 1)
                endInd = min(startInd + widInc, learn_nsteps)
                startInd = max(startInd, 1)
                u_rollave[startInd:endInd,:] .+= transpose(u)
                u_bale_rollave[startInd:endInd,:] .+= transpose(u_bale)
                u_bali_rollave[startInd:endInd,:] .+= transpose(u_bali)
                u_plas_rollave[startInd:endInd,:] .+= transpose(uX_plas)
                u_rollave_cnt[startInd:endInd,:] .+= 1
            end
        end

        # compute synapse-filtered spike trains
        @static if kind in [:train, :train_test]
            @static if typeof(p.tau_plas)<:Number
                axpby!(invtau_plas, spikesPrev, plusone-dt*invtau_plas, r)
            else
                r .+= (-dt.*r .+ spikesPrev) .* invtau_plas
            end
        end

        # apply external inputs
        if t > stim_on && t < stim_off
            X .= X_bal .+ X_stim[ti-round(Int,stim_on/dt),:,itask]
        else
            X .= X_bal
        end

        @static if p.sig>0 && p.noise_model==:voltage
            v .+= sig .* noise
        end

        # not in refractory period
        bnotrefrac .= t .> (lastSpike .+ refrac)
        @inline cellModel_timestep!(bnotrefrac, v, X, u, cellModel_args)

        # spike occurred
        @inline cellModel_spiked!(bspike, bnotrefrac, v, cellModel_args)
        @static kind in [:train, :train_test] && (spikes .= bspike)
        ns .+= bspike
        @inline cellModel_reset!(bspike, v, cellModel_args)
        lastSpike .= ifelse.(bspike, t, lastSpike)
        @static if kind in [:test, :train_test]
            times[ CartesianIndex.(1:Ncells, min.(maxTimes, bspike.*ns) .+ 1) ] .= ti
        end

        # accumulate the contribution of spikes to postsynaptic currents
        update_inputs(bspike,
                      w0Index, w0Weights, inputsE, inputsI,
                      wpIndexOut, wpWeightOut, inputsP)

        # external input to trained excitatory neurons
        @static p.LX>0 && if t > stim_off
            @static if kind in [:train, :train_test]
                @static if typeof(p.tau_plas)<:Number
                    axpby!(invtau_plas, spikesXPrev, plusone-dt*invtau_plas, rX)
                else
                    rX .+= (-dt.*rX .+ spikesXPrev) .* invtau_plas
                end
            end

            tidx = ti - round(Int, stim_off/dt)
            rand!(rng, rndX)
            # feed-forward neuron spiked
            bspikeX .= rndX .< @view rateX[tidx,:]
            @static kind in [:train, :train_test] && (spikesX .= bspikeX)
            nsX .+= bspikeX
            @static if kind in [:test, :train_test]
                timesX[ CartesianIndex.(1:LX, min.(maxTimes, bspikeX.*nsX) .+ 1) ] .= ti
            end
            inputsP[2:end] .+= dropdims(sum(view(wpWeightX,:,bspikeX), dims=2), dims=2)
        end

        # save spiking activities produced at the current time step
        @static if p.K>0
            inputsEPrev .= inputsE
            inputsIPrev .= inputsI
        end
        inputsPPrev .= inputsP
        @static if kind in [:train, :train_test]
            spikesPrev .= spikes
            @static p.LX>0 && (spikesXPrev .= spikesX)
        end
    end

    @static if kind in [:test, :train_test]
        u_rollave ./= u_rollave_cnt
        u_bale_rollave ./= u_rollave_cnt
        u_bali_rollave ./= u_rollave_cnt
        u_plas_rollave ./= u_rollave_cnt

        return ns, times[:,2:end], nsX, timesX[:,2:end],
               u_rollave, u_bale_rollave, u_bali_rollave, u_plas_rollave,
               u_exccell, u_inhcell, u_bale_exccell, u_bali_exccell,
               u_bale_inhcell, u_bali_inhcell, u_plas_exccell, u_plas_inhcell
    end

end
