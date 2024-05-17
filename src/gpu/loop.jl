function update_inputs(bspike,
                       w0Index, w0Weights, inputsE, inputsI,
                       wpIndexOut, wpWeightOut, inputsP,
                       charge0::T) where T

    function kernel(bspike, charge0,
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
                            CUDA.@atomic inputsE[0x1 + w0Index[j,i]] += max(w0Weights[j,i], charge0)
                            CUDA.@atomic inputsI[0x1 + w0Index[j,i]] += min(w0Weights[j,i], charge0)
                        end
                    end
                    if j<=size(wpIndexOut,1)
                        CUDA.@atomic inputsP[0x1 + wpIndexOut[j,i]] += wpWeightOut[j+1,i+1]
                    end
                end
            end
        end
        return nothing
    end

    if T<:Real
        _inputsE, _inputsI, _inputsP = inputsE, inputsI, inputsP
        _w0Weights, _wpWeightOut = w0Weights, wpWeightOut
        _charge0 = charge0
    else
        _inputsE, _inputsI, _inputsP = ustrip(inputsE), ustrip(inputsI), ustrip(inputsP)
        _w0Weights, _wpWeightOut = ustrip(w0Weights), ustrip(wpWeightOut)
        _charge0 = ustrip(charge0)
    end

    kernel = @cuda launch=false kernel(bspike, _charge0,
                                       w0Index, _w0Weights, _inputsE, _inputsI,
                                       wpIndexOut, _wpWeightOut, _inputsP)
    config = launch_configuration(kernel.fun)
    dims = (length(bspike),
            (@static p.K>0 ? max(size(w0Index,1),size(wpIndexOut,1)) : size(wpIndexOut,1)))
    xthreads = min(32, dims[1])
    ythreads = min(fld(config.threads, xthreads), cld(prod(dims), xthreads))
    xblocks = min(config.blocks, cld(dims[1], xthreads))
    yblocks = min(cld(config.blocks, xblocks), cld(dims[2], ythreads))
    kernel(bspike, _charge0,
           w0Index, _w0Weights, _inputsE, _inputsI,
           wpIndexOut, _wpWeightOut, _inputsP;
           threads=(xthreads,ythreads), blocks=(xblocks<<2,yblocks<<2))
end

function loop(::Val{Kind}, ::Type{TCurrent}, ::Type{TCharge}, ::Type{TTime},
              itask, learn_every, stim_on, stim_off, train_time, dt, Nsteps,
              Ncells, Ne, LX, refrac, learn_step, learn_nsteps, invtau_bale, invtau_bali,
              invtau_plas, X_bal, maxTimes, sig, wid, example_neurons,
              plusone, PScale, cellModel_args, bnotrefrac, bspike, bspikeX,
              scratch, raug, k, k2, rrXg, vPv, den, e, delta, rng, P, Pinv,
              X_stim, utarg,
              rateX, w0Index, w0Weights, wpWeightX, wpIndexIn, wpIndexOut,
              wpIndexConvert, wpWeightIn, wpWeightOut) where {Kind, TCurrent,
              TCharge, TTime}

    @unpack times, ns, timesX, nsX, inputsE, inputsI, inputsP, inputsEPrev, inputsIPrev, inputsPPrev, spikes, spikesPrev, spikesX, spikesXPrev, u_bale, u_bali, uX_plas, u_bal, u, r, rX, rrXhistory, X, lastSpike, v, noise, rndX, u_exccell, u_inhcell, u_bale_exccell, u_bali_exccell, u_bale_inhcell, u_bali_inhcell, u_plas_exccell, u_plas_inhcell, u_rollave, u_bale_rollave, u_bali_rollave, u_plas_rollave, u_rollave_cnt = scratch

    current0 = TCurrent(0)
    charge0 = TCharge(0)
    time0 = TTime(0)
    time1 = TTime(1)

    if Kind in (:test, :train_test)
        widInc = round(Int, 2*wid/learn_every - 1)

        u_exccell .= u_inhcell .= current0
        u_bale_exccell .= u_bali_exccell .= current0
        u_bale_inhcell .= u_bali_inhcell .= current0
        u_plas_exccell .= u_plas_inhcell .= current0

        u_rollave .= u_bale_rollave .= u_bali_rollave .= u_plas_rollave .= current0
        u_rollave_cnt .= 0
    end

    if Kind in (:train, :train_test)
        learn_seq = 1
        r .= 0/time1
        spikesPrev .= 0
        @static if p.LX>0
            spikesXPrev .= 0
            rX .= 0/time1
        end
    end

    if Kind in (:test, :train_test)
        times .= 0
        @static p.LX>0 && (timesX .= 0)
    end

    ns .= 0
    @static p.LX>0 && (nsX .= 0)
    lastSpike .= TTime(-100.0)
    cellModel_init!(v, rng, cellModel_args)
    @static if p.K>0
        u_bale .= u_bali .= current0
        inputsEPrev .= inputsIPrev .= charge0
    end
    uX_plas .= current0
    inputsPPrev .= charge0

    stim_on_steps = round(Int,stim_on/dt)
    @static p.LX>0 && (stim_off_steps =  round(Int, stim_off/dt))

    # start the actual training
    for ti=1:Nsteps
        t = dt*ti;

        # reset spiking activities from the previous time step
        @static p.K>0 && (inputsE .= inputsI .= charge0)
        inputsP .= charge0

        # modify the plastic weights when the stimulus is turned off 
        if Kind in (:train, :train_test)
            if t > stim_off && t <= train_time && mod(ti, learn_step) == 0
                disable_sigint() do
                    @static if p.PCompute == :fast
                        rls(itask,
                            raug, k, vPv, den, e, delta, r, rX,
                            P, u_bal, utarg, LX, learn_seq, wpIndexIn,
                            wpIndexConvert, wpWeightX, wpWeightIn, wpWeightOut,
                            plusone, exactlyzero, PScale)
                    else
                        rls(itask,
                            raug, k, k2, rrXg, e, delta, Ncells, p.PComputeN, r, rX,
                            Pinv, u_bal, utarg,
                            rrXhistory, charge0, LX, p.penmu, p.penlamFF, p.penlambda,
                            learn_seq, wpIndexIn,
                            wpIndexConvert, wpWeightX, wpWeightIn, wpWeightOut,
                            plusone, exactlyzero, PScale)
                    end
                end
                learn_seq += 1
            end
        end

        @static p.sig>0 && randn!(rng, TTime<:Real ? noise : ustrip(noise))
        u_bal .= current0

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

        if Kind in (:test, :train_test)
            # saved for visualization
            u_exccell[ti,:] .= u[1:example_neurons]
            u_bale_exccell[ti,:] .= u_bale[1:example_neurons]
            u_bali_exccell[ti,:] .= u_bali[1:example_neurons]
            u_plas_exccell[ti,:] .= uX_plas[1:example_neurons]
            u_inhcell[ti,:] .= u[1:example_neurons]
            u_bale_inhcell[ti,:] .= u_bale[end-example_neurons+1:end]
            u_bali_inhcell[ti,:] .= u_bali[end-example_neurons+1:end]
            u_plas_inhcell[ti,:] .= uX_plas[end-example_neurons+1:end]

            # save rolling average for analysis
            if t > stim_off && t <= train_time && mod(t, time1) == time0
                startInd = floor(Int, (t - stim_off - wid) / learn_every + 1)
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
        if Kind in (:train, :train_test)
            @static if typeof(p.tau_plas)<:Number
                axpby!(invtau_plas, spikesPrev, plusone-dt*invtau_plas, r)
            else
                r .+= (-dt.*r .+ spikesPrev) .* invtau_plas
            end
        end

        # apply external inputs
        if t > stim_on && t < stim_off
            X .= X_bal .+ X_stim[ti-stim_on_steps,:,itask]
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
        Kind in (:train, :train_test) && (spikes[2:end] .= bspike)
        ns .+= bspike
        @inline cellModel_reset!(bspike, v, cellModel_args)
        lastSpike .= ifelse.(bspike, t, lastSpike)
        if Kind in (:test, :train_test)
            times[ CartesianIndex.(1:Ncells, min.(maxTimes, bspike.*ns) .+ 1) ] .= ti
        end

        # accumulate the contribution of spikes to postsynaptic currents
        update_inputs(bspike,
                      w0Index, w0Weights, inputsE, inputsI,
                      wpIndexOut, wpWeightOut, inputsP,
                      charge0)

        # external input to trained excitatory neurons
        @static p.LX>0 && if t > stim_off
            if Kind in (:train, :train_test)
                @static if typeof(p.tau_plas)<:Number
                    axpby!(invtau_plas, spikesXPrev, plusone-dt*invtau_plas, rX)
                else
                    rX .+= (-dt.*rX .+ spikesXPrev) .* invtau_plas
                end
            end

            tidx = ti - stim_off_steps
            rand!(rng, rndX)
            # feed-forward neuron spiked
            bspikeX .= rndX .< @view rateX[tidx,:]
            Kind in (:train, :train_test) && (spikesX .= bspikeX)
            nsX .+= bspikeX
            if Kind in (:test, :train_test)
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
        if Kind in (:train, :train_test)
            spikesPrev .= spikes
            @static p.LX>0 && (spikesXPrev .= spikesX)
        end
    end

    if Kind in (:test, :train_test)
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
