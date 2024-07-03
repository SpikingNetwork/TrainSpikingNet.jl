function update_inputs(::Val{Kind}, Ncells, lastSpike, t, charge0,
                       w0Index, w0Weights, inputsE, inputsI,
                       wpIndexOut, wpWeightOut, inputsP) where {Kind}
    for ci = 1:Ncells
        if lastSpike[ci] == t
            # network connectivity is divided into two parts:
            #   - balanced connections (static)
            #   - plastic connections

            # (1) balanced connections (static)
            # loop over neurons (indexed by j) postsynaptic to neuron ci.
            @static p.K>0 && for j in eachindex(w0Index[ci])
                post_ci = w0Index[ci][j]         # cell index of j_th postsynaptic neuron
                wgt = w0Weights[ci][j]           # synaptic weight of the connection, ci -> post_ci
                if wgt > charge0                 # excitatory synapse
                    inputsE[post_ci] += wgt      #   - neuron ci spike's excitatory contribution to post_ci's synaptic current
                elseif wgt < charge0             # inhibitory synapse
                    inputsI[post_ci] += wgt      #   - neuron ci spike's inhibitory contribution to post_ci's synaptic current
                end
            end #end loop over synaptic projections

            # (2) plastic connections
            # loop over neurons (indexed by j) postsynaptic to neuron ci. 
            if Kind in (:train, :test, :train_test)
                for j in eachindex(wpIndexOut[ci])
                    post_ci = wpIndexOut[ci][j]               # cell index of j_th postsynaptic neuron
                    inputsP[post_ci] += wpWeightOut[ci][j]    # neuron ci spike's contribution to post_ci's synaptic current
                end
            end
        end
    end #end loop over neurons
end

function loop(::Val{Kind}, ::Type{TCurrent}, ::Type{TCharge}, ::Type{TTime},
              itask, learn_every, stim_on, stim_off, train_time, dt, Nsteps,
              u0_skip_steps, u0_ncells, Ncells, Ne, LX, refrac, learn_step, learn_nsteps,
              invtau_bale, invtau_bali, invtau_plas, X_bal, maxTimes,
              sig, wid, example_neurons, plusone, exactlyzero, PScale,
              cellModel_args, uavg, ustd, scratch, raug, k, rrXg, delta, rng, P, Pinv, work,
              lwork, pivot, X_stim, utarg, rateX, w0Index, w0Weights, wpIndexIn, wpIndexOut,
              wpIndexConvert, wpWeightX, wpWeightIn, wpWeightOut) where
              {Kind, TCurrent, TCharge, TTime}

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
    if Kind in (:train, :test, :train_test)
        uX_plas .= current0
        inputsPPrev .= charge0
    end

    if Kind in (:train, :test, :train_test)
        stim_on_steps = round(Int, stim_on/dt)
    end
    @static p.LX>0 && (stim_off_steps =  round(Int, stim_off/dt))

    # start the actual training
    for ti=1:Nsteps
        t = dt*ti;

        # reset spiking activities from the previous time step
        @static p.K>0 && (inputsE .= inputsI .= charge0)
        if Kind in (:train, :test, :train_test)
            inputsP .= charge0
        end
        if Kind in (:train, :train_test)
            spikes .= 0
            @static p.LX>0 && (spikesX .= 0)
        end

        # modify the plastic weights when the stimulus is turned off 
        #   - training occurs within the time interval [stim_off, train_time]
        #   - we need two versions of the plastic connectivity: 
        #       * one for learning (wpWeightIn) and 
        #       * the other for simulating network activity (wpWeightOut)
        #       * wpWeightIn and wpWeightOut represent the same underlying plastic connectivity
        #   - wpWeightIn is updated every learn_every milliseconds by the recursive least squares algorithm
        #   - convertWgtIn2Out() converts wpWeightIn to wpWeightOut at the end of rls()
        #
        # wpWeightIn: - plastic weights used for and modified by the RLS training algorithm 
        #             - length Ncell vector of length Kin = Lexc + Linh vectors
        #             - wpWeightIn[i]:
        #                 - weights of the incoming connections to neuron i
        #                 - each i updated independently by the RLS algorithm.
        # wpIndexIn:  - length Ncell vector of length Kin = Lexc + Linh vectors
        #             - wpIndexIn[i]:
        #                 - Indices of presynaptic neurons that connect to neuron i
        #                 - Fixed throughout the simulation
        # wpWeightOut: - plastic weights used for simulating network activities
        #              - length Ncell vector of length Kout vectors, where Kout is the number of outgoing plastic synapses from each neuron
        #              - wpWeightOut[i]:
        #                  - weights of the outgoing connections from neuron i
        #                  - Used to compute inputsP
        # wpIndexOut:  - length Ncell vector of length Kout vectors
        #              - wpIndexOut[i]:
        #                  - Indices of postsynaptic neurons that neuron i connects to
        #                  - Fixed throughout the simulation. Used to compute inputsP

        if Kind in (:train, :train_test)
            if t > stim_off && t <= train_time && mod(ti, learn_step) == 0
                disable_sigint() do
                    @static if p.PCompute == :fast
                        rls(itask,
                            raug, k, delta, Ncells, r, rX, P,
                            u_bal, utarg, LX, learn_seq, wpIndexIn,
                            wpIndexConvert, wpWeightX, wpWeightIn, wpWeightOut,
                            plusone, exactlyzero, PScale)
                    else
                        rls(itask,
                            raug, k, rrXg, delta, Ncells, r, rX, Pinv, work, lwork, pivot,
                            u_bal, utarg,
                            rrXhistory, charge0, LX, p.penmu, p.penlamFF, p.penlambda,
                            learn_seq, wpIndexIn,
                            wpIndexConvert, wpWeightX, wpWeightIn, wpWeightOut,
                            plusone, exactlyzero, PScale)
                    end
                end
                learn_seq += 1
            end
        end

        if Kind in (:test, :train_test)
            if t > stim_off && t <= train_time && mod(t, time1) == time0
                startInd = floor(Int, (t - stim_off - wid) / learn_every + 1)
                endInd = min(startInd + widInc, learn_nsteps)
                startInd = max(startInd, 1)
                u_rollave_cnt[startInd:endInd] .+= 1
            end
        end

        @static p.sig>0 && randn!(rng, TTime<:Real ? noise : ustrip(noise))
        u_bal .= current0

        # update network activities:
        #   - synaptic currents (u_bale, u_bali, uX_plas)
        #   - membrane potential (v) 
        #   - weighted spikes received by each neuron (inputsE, inputsI, inputsP)
        #   - activity variables used for training
        #       * spikes emitted by each neuron (spikes)
        #       * synapse-filtered spikes emitted by each neuron (r)        
        @maybethread :dynamic for ci = 1:Ncells
            @static if p.K>0
                u_bale[ci] += (-dt*u_bale[ci] + inputsEPrev[ci]) * invtau_bale
                u_bali[ci] += (-dt*u_bali[ci] + inputsIPrev[ci]) * invtau_bali
            end
            if Kind in (:train, :test, :train_test)
                @static if typeof(p.tau_plas)<:Number
                    uX_plas[ci] += (-dt*uX_plas[ci] + inputsPPrev[ci]) * invtau_plas
                else
                    uX_plas[ci] += (-dt*uX_plas[ci] + inputsPPrev[ci]) * invtau_plas[ci]
                end
            end

            @static p.K>0 && (u_bal[ci] += u_bale[ci] + u_bali[ci])
            @static if p.sig>0 && p.noise_model==:current
                u_bal[ci] += sig[ci] * noise[ci]
            end
            u[ci] = u_bal[ci]
            if Kind in (:train, :test, :train_test)
                u[ci] += uX_plas[ci]
            end

            if Kind == :init
                if ti > u0_skip_steps 
                    uavg[ci] += u[ci]
                    if ci <= u0_ncells
                        ustd[ti - u0_skip_steps, ci] = u[ci]
                    end
                end
            end

            if Kind in (:test, :train_test)
                # saved for visualization
                if ci <= example_neurons
                    u_exccell[ti,ci] = u[ci]
                    u_bale_exccell[ti,ci] = u_bale[ci]
                    u_bali_exccell[ti,ci] = u_bali[ci]
                    u_plas_exccell[ti,ci] = uX_plas[ci]
                elseif ci >= Ncells - example_neurons + 1
                    u_inhcell[ti,ci-Ncells+example_neurons] = u[ci]
                    u_bale_inhcell[ti,ci-Ncells+example_neurons] = u_bale[ci]
                    u_bali_inhcell[ti,ci-Ncells+example_neurons] = u_bali[ci]
                    u_plas_inhcell[ti,ci-Ncells+example_neurons] = uX_plas[ci]
                end

                # save rolling average for analysis
                if t > stim_off && t <= train_time && mod(t, time1) == time0
                    u_rollave[startInd:endInd,ci] .+= u[ci]
                    u_bale_rollave[startInd:endInd,ci] .+= u_bale[ci]
                    u_bali_rollave[startInd:endInd,ci] .+= u_bali[ci]
                    u_plas_rollave[startInd:endInd,ci] .+= uX_plas[ci]
                end
            end

            # compute synapse-filtered spike trains
            if Kind in (:train, :train_test)
                @static if typeof(p.tau_plas)<:Number
                    r[ci] += (-dt*r[ci] + spikesPrev[ci]) * invtau_plas
                else
                    r[ci] += (-dt*r[ci] + spikesPrev[ci]) * invtau_plas[ci]
                end
            end

            # apply external inputs
            #   - X_bal: default inputs to maintain the balanced state
            #   - X_stim: inputs that trigger the learned responses,
            #           applied within the time interval [stim_on, stim_off]
            if Kind in (:train, :test, :train_test)
                if t > stim_on && t < stim_off
                    X[ci] = X_bal[ci] + X_stim[ti-stim_on_steps,ci,itask]
                else
                    X[ci] = X_bal[ci]
                end
            end
            Kind == :init && (X[ci] = X_bal[ci])

            @static if p.sig>0 && p.noise_model==:voltage
                v[ci] += sig[ci] * noise[ci]
            end

            # not in refractory period
            if t > (lastSpike[ci] + refrac)  
                # update membrane potential
                @inline cellModel_timestep!(ci, v, X, u, cellModel_args)

                #spike occurred
                if @inline cellModel_spiked(ci, v, cellModel_args)
                    @inline cellModel_reset!(ci, v, cellModel_args)  # reset voltage
                    if Kind in (:train, :train_test)
                        spikes[ci] = 1  # record that neuron ci spiked. Used for computing r[ci]
                    end
                    lastSpike[ci] = t  # record neuron ci's last spike time. Used for checking ci is not in refractory period
                    ns[ci] += 1  # number of spikes neuron ci emitted
                    if Kind in (:test, :train_test)
                        ns[ci] <= maxTimes && (times[ci, ns[ci]] = ti)
                    end
                end #end if(spike occurred)
            end #end not in refractory period
        end #end loop over neurons

        # accumulate the contribution of spikes to postsynaptic currents
        update_inputs(Val(Kind), Ncells, lastSpike, t, charge0,
                      w0Index, w0Weights, inputsE, inputsI, wpIndexOut, wpWeightOut, inputsP)

        # external input to trained excitatory neurons
        #   - rX : filtered spike trains of feed-forward spikes
        #   - rateX : pre-defined spiking rate of external neurons
        #   - spikesX
        #   - spikesXPrev
        # (1) simulation: feed-forward spikes are added to inputsP
        # (2) training: spikesXPrev computes the filtered feed-forward spikes, rX
        @static p.LX>0 && if t > stim_off
            if Kind in (:train, :train_test)
                @maybethread :dynamic for ci = 1:LX
                    # if training, filter the spikes
                    @static if typeof(p.tau_plas)<:Number
                        rX[ci] += (-dt*rX[ci] + spikesXPrev[ci])*invtau_plas
                    else
                        rX[ci] += (-dt*rX[ci] + spikesXPrev[ci])*invtau_plas[ci]
                    end
                end
            end

            tidx = ti - stim_off_steps
            rand!(rng, rndX)
            for ci = 1:LX
                # feed-forward neuron spiked
                if rndX[ci] < rateX[tidx,ci]
                    Kind in (:train, :train_test) && (spikesX[ci] = 1)
                    nsX[ci] += 1
                    if Kind in (:test, :train_test)
                        nsX[ci] <= maxTimes && (timesX[ci, nsX[ci]] = ti)
                    end
                    if Kind in (:train, :test, :train_test)
                        inputsP += @view wpWeightX[:,ci]
                    end
                end #end if spiked
            end #end loop over feed-forward neurons
        end #end feed-forward input

        # save spiking activities produced at the current time step
        #   - inputsPrev's will be used in the next time step to compute synaptic currents (u_bale, u_bali, uX_plas)
        #   - spikesPrev will be used in the next time step to compute synapse-filter spikes (r)
        @static if p.K>0
            inputsEPrev .= inputsE
            inputsIPrev .= inputsI
        end
        Kind in (:train, :test, :train_test) && (inputsPPrev .= inputsP)
        if Kind in (:train, :train_test)
            spikesPrev .= spikes
            @static p.LX>0 && (spikesXPrev .= spikesX)
        end

    end #end loop over time

    if Kind == :init
        mean_exc_rate = mean(ns[1:Ne]) / train_time
        mean_inh_rate = mean(ns[Ne+1:Ncells]) / train_time
        if TTime <: Real
            mean_exc_rate_conv = string(1000*mean_exc_rate, " Hz")
            mean_inh_rate_conv = string(1000*mean_inh_rate, " Hz")
        else
            mean_exc_rate_conv = uconvert(HzUnit, mean_exc_rate)
            mean_inh_rate_conv = uconvert(HzUnit, mean_inh_rate)
        end
        println("mean excitatory firing rate: ", mean_exc_rate_conv)
        println("mean inhibitory firing rate: ", mean_inh_rate_conv)
                    
        return uavg ./ (Nsteps - u0_skip_steps), ns, mean(std(ustd, dims=1))
    end

    if Kind in (:test, :train_test)
        u_rollave ./= u_rollave_cnt
        u_bale_rollave ./= u_rollave_cnt
        u_bali_rollave ./= u_rollave_cnt
        u_plas_rollave ./= u_rollave_cnt

        return ns, times, nsX, timesX,
               u_rollave, u_bale_rollave, u_bali_rollave, u_plas_rollave,
               u_exccell, u_inhcell, u_bale_exccell, u_bali_exccell,
               u_bale_inhcell, u_bali_inhcell, u_plas_exccell, u_plas_inhcell
    end

end #end function
