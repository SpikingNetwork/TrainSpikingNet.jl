@eval function $(Symbol("loop_",kind))(itask,
    learn_every, stim_on, stim_off, train_time, dt, Nsteps, Ncells, Ne,
    Lei, LX, refrac, vre, invtau_bale, invtau_bali, invtau_plas, X_bal,
    thresh, tau_mem, maxTimes, times, ns, timesX, nsX, inputsE,
    inputsI, inputsP, inputsEPrev, inputsIPrev, inputsPPrev, spikes,
    spikesPrev, spikesX, spikesXPrev, u_bale, u_bali, uX_plas,
    u_bal, u, r, rX, X, wid, example_neurons, lastSpike,
    plusone, exactlyzero, PScale, raug, k, v, rng, noise, rndX, sig,
    P, Px, w0Index, w0Weights, nc0, X_stim, utarg, wpIndexIn, wpIndexOut,
    wpIndexConvert, wpWeightX, wpWeightIn, wpWeightOut, ncpIn, ncpOut,
    uavg, utmp, rateX)

    @static kind in [:init, :train, :train_test] && (steps_per_sec = round(Int, 1000/dt))

    @static kind in [:train, :train_test] && Param.LX>0 && (rateX /= steps_per_sec)

    @static if kind in [:test, :train_test]
        learn_nsteps = round(Int, (train_time - stim_off)/learn_every)
        widInc = round(Int, 2*wid/learn_every - 1)
                
        u_exccell = zeros(eltype(u), Nsteps,example_neurons)
        u_inhcell = zeros(eltype(u), Nsteps,example_neurons)
        u_bale_exccell = zeros(eltype(u), Nsteps,example_neurons)
        u_bali_exccell = zeros(eltype(u), Nsteps,example_neurons)
        u_bale_inhcell = zeros(eltype(u), Nsteps,example_neurons)
        u_bali_inhcell = zeros(eltype(u), Nsteps,example_neurons)
        u_plas_exccell = zeros(eltype(u), Nsteps,example_neurons)
        u_plas_inhcell = zeros(eltype(u), Nsteps,example_neurons)

        u_rollave = zeros(eltype(u), learn_nsteps,Ncells)
        u_bale_rollave = zeros(eltype(u), learn_nsteps,Ncells)
        u_bali_rollave = zeros(eltype(u), learn_nsteps,Ncells)
        u_plas_rollave = zeros(eltype(u), learn_nsteps,Ncells)
        u_rollave_cnt = zeros(Int, learn_nsteps)
    end

    @static if kind in [:train, :train_test]
        learn_seq = 1
        r .= 0
        spikesPrev .= 0
        @static if Param.LX>0
            spikesXPrev .= 0
            rX .= 0
        end
    end

    @static if kind in [:test, :train_test]
        times .= 0
        @static Param.LX>0 && (timesX .= 0)
    end

    ns .= 0
    @static Param.LX>0 && (nsX .= 0)
    lastSpike .= -100.0
    randn!(rng, v)
    @static if Param.K>0
        u_bale .= u_bali .= 0
        inputsEPrev .= inputsIPrev .= 0.0
    end
    @static if kind in [:train, :test, :train_test]
        uX_plas .= 0
        inputsPPrev .= 0.0
    end
    @static if Param.sig>0
        @static if Param.noise_model==:voltage
            sqrtdt = sqrt(dt)
            sqrtinvtau_mem = sqrt.(1 ./ tau_mem)
        elseif Param.noise_model==:current
            invsqrtdt = 1/sqrt(dt)
            sqrttau_mem = sqrt.(tau_mem)
        end
    end
    invtau_mem = 1 ./ tau_mem

    # start the actual training
    for ti=1:Nsteps
        t = dt*ti;

        # reset spiking activities from the previous time step
        @static Param.K>0 && (inputsE .= inputsI .= 0.0)
        @static kind in [:train, :test, :train_test] && (inputsP .= 0.0)
        @static if kind in [:train, :train_test]
            spikes .= 0.0
            @static Param.LX>0 && (spikesX .= 0.0)
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
        #             - Kin x Ncell matrix where Kin = Lexc + Linh is the number of incoming plastic synapses to each neuron
        #             - ith col, wpWeightIn[:,i]:
        #                 - weights of the incoming connections to neuron i
        #                 - each row of wpWeightIn will be updated independently by the RLS algorithm.
        # wpIndexIn:  - Ncell x Kin matrix where Kin = Lexc + Linh
        #             - ith row, wpIndexIn[i,:]:
        #                 - Indices of presynaptic neurons that connect to neuron i
        #                 - Fixed throughout the simulation. Used to define Px
        # wpWeightOut: - plastic weights used for simulating network activities
        #              - Kout x Ncell matrix where Kout is the number of outgoing plastic synapses from each neuron
        #              - the actual number of outgoing plastic synapses is different across neurons, so we chose a fixed number Kout >= Lexc + Linh
        #              - ith column, wpWeightOut[:,i]:
        #                  - weights of the outgoing connections from neuron i
        #                  - Used to compute inputsP
        # wpIndexOut:  - Kout x Ncell matrix
        #              - ith column, wpIndexOut[:,i]:
        #                  - Indices of postsynaptic neurons that neuron i connect to
        #                  - Fixed throughout the simulation. Used to compute inputsP

        @static kind in [:train, :train_test] && if t > stim_off && t <= train_time && mod(ti, learn_step) == 0
            wpWeightIn, wpWeightOut = rls(itask,
                    raug, k, delta, Ncells, Lei, r, rX, Px, P,
                    u_bal, utarg, learn_seq, ncpIn, wpIndexIn,
                    wpIndexConvert, wpWeightX, wpWeightIn, wpWeightOut,
                    plusone, exactlyzero)
            learn_seq += 1
        end

        @static kind in [:test, :train_test] && if t > stim_off && t <= train_time && mod(t,1.0) == 0
            startInd = floor(Int, (t - stim_off - wid)/learn_every + 1)
            endInd = min(startInd + widInc, learn_nsteps)
            startInd = max(startInd, 1)
            u_rollave_cnt[startInd:endInd] .+= 1
        end

        @static Param.sig>0 && randn!(rng, noise)
        u_bal .= 0.0

        # update network activities:
        #   - synaptic currents (u_bale, u_bali, uX_plas)
        #   - membrane potential (v) 
        #   - weighted spikes received by each neuron (inputsE, inputsI, inputsP)
        #   - activity variables used for training
        #       * spikes emitted by each neuron (spikes)
        #       * synapse-filtered spikes emitted by each neuron (r)        
        @maybethread for ci = 1:Ncells
            @static if Param.K>0
                u_bale[ci] += (-dt*u_bale[ci] + inputsEPrev[ci]) * invtau_bale
                u_bali[ci] += (-dt*u_bali[ci] + inputsIPrev[ci]) * invtau_bali
            end
            @static if kind in [:train, :test, :train_test]
                @static if typeof(Param.tau_plas)<:Number
                    uX_plas[ci] += (-dt*uX_plas[ci] + inputsPPrev[ci]) * invtau_plas
                else
                    uX_plas[ci] += (-dt*uX_plas[ci] + inputsPPrev[ci]) * invtau_plas[ci]
                end
            end

            @static Param.K>0 && (u_bal[ci] += u_bale[ci] + u_bali[ci])
            @static if Param.sig>0 && Param.noise_model==:current
                u_bal[ci] += invsqrtdt * sqrttau_mem[ci] * sig[ci] * noise[ci]
            end
            u[ci] = u_bal[ci]
            @static if kind in [:train, :test, :train_test]
                u[ci] += uX_plas[ci]
            end

            @static if kind == :init
                if ti > 1000/dt # 1000 ms
                    uavg[ci] += u[ci] # save u
                    if ci <= 1000
                        utmp[ti - steps_per_sec, ci] = u[ci]
                    end
                end
            end

            @static if kind in [:test, :train_test]
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
                if t > stim_off && t <= train_time && mod(t,1.0) == 0
                    u_rollave[startInd:endInd,ci] .+= u[ci]
                    u_bale_rollave[startInd:endInd,ci] .+= u_bale[ci]
                    u_bali_rollave[startInd:endInd,ci] .+= u_bali[ci]
                    u_plas_rollave[startInd:endInd,ci] .+= uX_plas[ci]
                end
            end

            # compute synapse-filtered spike trains
            @static if kind in [:train, :train_test]
                @static if typeof(Param.tau_plas)<:Number
                    r[ci] += (-dt*r[ci] + spikesPrev[ci]) * invtau_plas
                else
                    r[ci] += (-dt*r[ci] + spikesPrev[ci]) * invtau_plas[ci]
                end
            end

            # apply external inputs
            #   - X_bal: default inputs to maintain the balanced state
            #   - X_stim: inputs that trigger the learned responses,
            #           applied within the time interval [stim_on, stim_off]
            @static if kind in [:train, :test, :train_test]
                if t > stim_on && t < stim_off
                    X[ci] = X_bal[ci] + X_stim[ti-round(Int,stim_on/dt),ci,itask]
                else
                    X[ci] = X_bal[ci]
                end
            end
            @static kind == :init && (X[ci] = X_bal[ci])

            @static if Param.sig>0 && Param.noise_model==:voltage
                v[ci] += sqrtdt * sqrtinvtau_mem[ci] * sig[ci] * noise[ci]
            end

            # not in refractory period
            if t > (lastSpike[ci] + refrac)  
                # update membrane potential
                v[ci] += dt * invtau_mem[ci] * (X[ci] - v[ci] + u[ci])

                #spike occurred
                if v[ci] > thresh[ci]                      
                    v[ci] = vre                 # reset voltage
                    @static kind in [:train, :train_test] && (spikes[ci] = 1)   # record that neuron ci spiked. Used for computing r[ci]
                    lastSpike[ci] = t           # record neuron ci's last spike time. Used for checking ci is not in refractory period
                    ns[ci] += 1           # number of spikes neuron ci emitted
                    @static kind in [:test, :train_test] && if ns[ci] <= maxTimes
                        times[ci,ns[ci]] = t
                    end
                end #end if(spike occurred)
            end #end not in refractory period
        end #end loop over neurons

        # accumulate the contribution of spikes to postsynaptic currents
        for ci = 1:Ncells
            if lastSpike[ci] == t
                # network connectivity is divided into two parts:
                #   - balanced connections (static) 
                #   - plastic connections

                # (1) balanced connections (static)
                # loop over neurons (indexed by j) postsynaptic to neuron ci.                     
                # nc0[ci] is the number neurons postsynaptic neuron ci
                @static Param.K>0 && for j = 1:nc0[ci]                       
                    post_ci = w0Index[j,ci]                 # cell index of j_th postsynaptic neuron
                    wgt = w0Weights[j,ci]                   # synaptic weight of the connection, ci -> post_ci
                    if wgt > 0                              # excitatory synapse
                        inputsE[post_ci] += wgt      #   - neuron ci spike's excitatory contribution to post_ci's synaptic current
                    elseif wgt < 0                          # inhibitory synapse
                        inputsI[post_ci] += wgt      #   - neuron ci spike's inhibitory contribution to post_ci's synaptic current
                    end
                end #end loop over synaptic projections

                # (2) plastic connections
                # loop over neurons (indexed by j) postsynaptic to neuron ci. 
                # ncpOut[ci] is the number neurons postsynaptic neuron ci
                @static kind in [:train, :test, :train_test] && for j = 1:ncpOut[ci]
                    post_ci = wpIndexOut[j,ci]                 # cell index of j_th postsynaptic neuron
                    inputsP[post_ci] += wpWeightOut[j,ci]    # neuron ci spike's contribution to post_ci's synaptic current
                end
            end
        end #end loop over neurons

        # external input to trained excitatory neurons
        #   - rX : filtered spike trains of feed-forward spikes
        #   - rateX : pre-defined spiking rate of external neurons
        #   - spikesX
        #   - spikesXPrev
        # (1) simulation: feed-forward spikes are added to inputsP
        # (2) training: spikesXPrev computes the filtered feed-forward spikes, rX
        @static Param.LX>0 && if t > stim_off
            @static if kind in [:train, :train_test]
                for ci = 1:LX
                    # if training, filter the spikes
                    @static if typeof(Param.tau_plas)<:Number
                        rX[ci] += (-dt*rX[ci] + spikesXPrev[ci])*invtau_plas
                    else
                        rX[ci] += (-dt*rX[ci] + spikesXPrev[ci])*invtau_plas[ci]
                    end
                end
            end

            tidx = ti - round(Int, stim_off/dt)
            rand!(rng, rndX)
            for ci = 1:LX
                # feed-forward neuron spiked
                if rndX[ci] < rateX[tidx,ci]
                    @static kind in [:train, :train_test] && (spikesX[ci] = 1)
                    nsX[ci] += 1
                    @static kind in [:test, :train_test] && if nsX[ci] <= maxTimes
                        timesX[ci,nsX[ci]] = t
                    end
                    @static if kind in [:train, :test, :train_test]
                        inputsP += @view wpWeightX[:,ci]
                    end
                end #end if spiked
            end #end loop over feed-forward neurons
        end #end feed-forward input

        # save spiking activities produced at the current time step
        #   - inputsPrev's will be used in the next time step to compute synaptic currents (u_bale, u_bali, uX_plas)
        #   - spikesPrev will be used in the next time step to compute synapse-filter spikes (r)
        @static if Param.K>0
            inputsEPrev .= inputsE
            inputsIPrev .= inputsI
        end
        @static kind in [:train, :test, :train_test] && (inputsPPrev .= inputsP)
        @static if kind in [:train, :train_test]
            spikesPrev .= spikes
            @static Param.LX>0 && (spikesXPrev .= spikesX)
        end

    end #end loop over time

    @static if kind == :init
        println("mean excitatory firing rate: ", 1000*mean(ns[1:Ne])/train_time, " Hz")
        println("mean inhibitory firing rate: ", 1000*mean(ns[(Ne+1):Ncells])/train_time, " Hz")
                    
        return uavg ./ (Nsteps - steps_per_sec), ns, mean(std(utmp, dims=1))
    end

    @static if kind in [:test, :train_test]
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
