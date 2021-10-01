@eval function $(Symbol("loop_",kind))(learn_every, stim_on, stim_off,
    train_time, dt, Nsteps, Ncells, Ne, refrac, vre, invtauedecay,
    invtauidecay, invtaudecay_plastic, mu, thresh, invtau, maxTimes, times,
    ns, forwardInputsE, forwardInputsI, forwardInputsP, forwardInputsEPrev,
    forwardInputsIPrev, forwardInputsPPrev, forwardSpike, forwardSpikePrev,
    xedecay, xidecay, xpdecay, synInputBalanced, synInput, r, bias, wid,
    example_neurons, lastSpike, plusone, k, v, rng, noise, sig, P, Px, w0Index,
    w0Weights, nc0, stim, xtarg, wpIndexIn, wpIndexOut, wpIndexConvert,
    wpWeightIn, wpWeightOut, ncpIn, ncpOut, uavg, utmp)

@static if kind in [:test, :train_test]
    learn_nsteps = round(Int, (train_time - stim_off)/learn_every)
    widInc = round(Int, 2*wid/learn_every - 1)
            
    vtotal_exccell = zeros(Nsteps,example_neurons)
    vtotal_inhcell = zeros(Nsteps,example_neurons)
    vebal_exccell = zeros(Nsteps,example_neurons)
    vibal_exccell = zeros(Nsteps,example_neurons)
    vebal_inhcell = zeros(Nsteps,example_neurons)
    vibal_inhcell = zeros(Nsteps,example_neurons)
    vplastic_exccell = zeros(Nsteps,example_neurons)
    vplastic_inhcell = zeros(Nsteps,example_neurons)

    xtotal = zeros(learn_nsteps,Ncells)
    xebal = zeros(learn_nsteps,Ncells)
    xibal = zeros(learn_nsteps,Ncells)
    xplastic = zeros(learn_nsteps,Ncells)
    xtotalcnt = zeros(Int, learn_nsteps)
    xebalcnt = zeros(Int, learn_nsteps)
    xibalcnt = zeros(Int, learn_nsteps)
    xplasticcnt = zeros(Int, learn_nsteps)
end

@static if kind in [:train, :train_test]
    learn_seq = 1
    r .= 0
end

ns .= 0
lastSpike .= -100.0
randn!(rng, v)
xedecay .= xidecay .= 0
@static if p.K>0
    forwardInputsEPrev .= forwardInputsIPrev .= 0.0
elseif kind in [:train, :test, :train_test]
    synInputBalanced .= 0.0
end
@static if kind in [:train, :test, :train_test]
    xpdecay .= 0
    forwardInputsPPrev .= 0.0
end
@static kind == :init && p.K==0 && (synInput .= 0.0)
@static p.K==0 && (sqrtdt = sqrt(dt))

# start the actual training
for ti=1:Nsteps
    t = dt*ti;

    # reset spiking activities from the previous time step
    @static p.K>0 && (forwardInputsE .= forwardInputsI .= 0.0)
    @static kind in [:train, :test, :train_test] && (forwardInputsP .= 0.0;)
    @static kind in [:train, :train_test] && (forwardSpike .= 0.0;)

    # start training the plastic weights when the stimulus is turned off 
    #   - training occurs within the time interval [stim_off, train_time]
    #   - we need two versions of the plastic connectivity: 
    #       * one for learning (wpWeightIn) and 
    #       * the other for simulating network activity (wpWeightOut)
    #       * wpWeightIn and wpWeightOut represent the same underlying plastic connectivity
    #   - wpWeightIn is updated every learn_every (=10ms) by the recursive least squares algorithm
    #   - convertWgtIn2Out() converts wpWeightIn to wpWeightOut at the end of rls
    #   - wpWeightOut is used for simulating the network activity
    #
    # wpWeightIn: - plastic weights used for and modified by the rls training algorithm 
    #             - Ncell x Kin matrix where Kin = Lexc + Linh is the number of incoming plastic synapses to each neuron
    #             - ith row, wpWeightIn[i,:]:
    #                 - weights of the incoming connections to neuron i
    #                 - each row of wpWeightIn will be updated independently by the rls algorithm. (see line 153)
    # wpIndexIn:  - Ncell x Kin matrix where Kin = Lexc + Linh
    #             - ith row, wpIndexIn[i,:]:
    #                 - Indices of presynaptic neurons that connect to neuron i
    #                 - Fixed throughout the simulation. Used to define Px (see line 83)
    # wpWeightOut: - plastic weights used for simulating network activities
    #              - Kout x Ncell matrix where Kout is the number of outgoing plastic synapses from each neuron
    #              - the actual number of outgoing plastic synapses is different across neurons, so we chose a fixed number Kout >= Lexc + Linh
    #              - ith column, wpWeightOut[:,i]:
    #                  - weights of the outgoing connections from neuron i
    #                  - Used to compute forwardInputsP (see line 221)
    # wpIndexOut:  - Kout x Ncell matrix
    #              - ith column, wpIndexOut[:,i]:
    #                  - Indices of postsynaptic neurons that neuron i connect to
    #                  - Fixed throughout the simulation. Used to compute forwardInputsP (see line 221)

    @static kind in [:train, :train_test] && if t > stim_off && t <= train_time && mod(ti, learn_step) == 0
        wpWeightIn, wpWeightOut = rls(k, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone)
        learn_seq += 1
    end

    @static kind in [:test, :train_test] && if t > stim_off && t <= train_time && mod(t,1.0) == 0
        startInd = floor(Int, (t - stim_off - wid)/learn_every + 1)
        endInd = min(startInd + widInc, learn_nsteps)
        startInd = max(startInd, 1)
        xtotalcnt[startInd:endInd] .+= 1
        xebalcnt[startInd:endInd] .+= 1
        xibalcnt[startInd:endInd] .+= 1
        xplasticcnt[startInd:endInd] .+= 1
    end

    @static p.K==0 && randn!(rng, noise)

    # update network activities:
    #   - synaptic currents (xedecay, xidecay, xpdecay)
    #   - membrane potential (v) 
    #   - weighted spikes received by each neuron (forwardInputsE, forwardInputsI, forwardInputsP)
    #   - activity variables used for training
    #       * spikes emitted by each neuron (forwardSpike)
    #       * synapse-filtered spikes emitted by each neuron (r)        
    @maybethread for ci = 1:Ncells
        @static if p.K>0
            xedecay[ci] += (-dt*xedecay[ci] + forwardInputsEPrev[ci]) * invtauedecay
            xidecay[ci] += (-dt*xidecay[ci] + forwardInputsIPrev[ci]) * invtauidecay
        end
        @static if kind in [:train, :test, :train_test]
            xpdecay[ci] += (-dt*xpdecay[ci] + forwardInputsPPrev[ci]) * invtaudecay_plastic
        end

        @static if kind in [:train, :test, :train_test]
            @static if p.K>0
                synInputBalanced[ci] = xedecay[ci] + xidecay[ci]
                synInput[ci] = synInputBalanced[ci] + xpdecay[ci]
            else
                synInput[ci] = xpdecay[ci]
            end
        end
        @static if kind == :init && p.K>0
            synInput[ci] = xedecay[ci] + xidecay[ci]
        end

        @static if kind == :init
            if ti > 1000/dt # 1000 ms
                uavg[ci] += synInput[ci] / (Nsteps - round(Int,1000/dt)) # save synInput
            end

            if ti > 1000/dt && ci <= 1000
                utmp[ti - round(Int,1000/dt), ci] = synInput[ci]
            end
        end

        @static if kind in [:test, :train_test]
            # saved for visualization
            if ci <= example_neurons
                vtotal_exccell[ti,ci] = synInput[ci]
                vebal_exccell[ti,ci] = xedecay[ci]
                vibal_exccell[ti,ci] = xidecay[ci]
                vplastic_exccell[ti,ci] = xpdecay[ci]
            elseif ci >= Ncells - example_neurons + 1
                vtotal_inhcell[ti,ci-Ncells+example_neurons] = synInput[ci]
                vebal_inhcell[ti,ci-Ncells+example_neurons] = xedecay[ci]
                vibal_inhcell[ti,ci-Ncells+example_neurons] = xidecay[ci]
                vplastic_inhcell[ti,ci-Ncells+example_neurons] = xpdecay[ci]
            end

            # save rolling average for analysis
            if t > stim_off && t <= train_time && mod(t,1.0) == 0
                xtotal[startInd:endInd,ci] .+= synInput[ci]
                xebal[startInd:endInd,ci] .+= xedecay[ci]
                xibal[startInd:endInd,ci] .+= xidecay[ci]
                xplastic[startInd:endInd,ci] .+= xpdecay[ci]
            end
        end

        # if training, compute synapse-filtered spike trains
        @static if kind in [:train, :train_test]
            r[ci] += (-dt*r[ci] + forwardSpikePrev[ci])*invtaudecay_plastic
        end

        # external inputs
        #   - mu: default inputs to maintain the balanced state
        #   - stim: inputs that trigger the learned responses
        #         : applied within the time interval [stim_on, stim_off]
        @static if kind in [:train, :test, :train_test]
            if t > stim_on && t < stim_off
                bias[ci] = mu[ci] + stim[ti-round(Int,stim_on/dt),ci]
            else
                bias[ci] = mu[ci]
            end
        end
        @static if kind == :init
            bias[ci] = mu[ci]
        end

        @static if p.K==0
            v[ci] += sqrtdt*sig[ci]*noise[ci]
        end

        # neuron ci not in refractory period
        if t > (lastSpike[ci] + refrac)  
            # update membrane potential
            v[ci] += dt*invtau[ci]*(bias[ci]-v[ci] + synInput[ci])

            #spike occurred
            if v[ci] > thresh[ci]                      
                v[ci] = vre                 # reset voltage
                @static kind in [:train, :train_test] && (forwardSpike[ci] = 1.)   # record that neuron ci spiked. Used for computing r[ci]
                lastSpike[ci] = t           # record neuron ci's last spike time. Used for checking ci is not in refractory period
                ns[ci] = ns[ci]+1           # number of spikes neuron ci emitted
                @static if kind in [:test, :train_test]
                    if ns[ci] <= maxTimes
                        times[ci,ns[ci]] = t
                    end
                end
            end #end if(spike occurred)
        end #end not in refractory period
    end #end loop over neurons

    for ci = 1:Ncells
        if lastSpike[ci] == t
            # Accumulate the contribution of spikes to postsynaptic currents
            # Network connectivity is divided into two parts:
            #   - balanced connections (static) 
            #   - plastic connections

            # (1) balanced connections (static)
            # loop over neurons (indexed by j) postsynaptic to neuron ci.                     
            # nc0[ci] is the number neurons postsynaptic neuron ci
            @static p.K>0 && for j = 1:nc0[ci]                       
                post_ci = w0Index[j,ci]                 # cell index of j_th postsynaptic neuron
                wgt = w0Weights[j,ci]                   # synaptic weight of the connection, ci -> post_ci
                if wgt > 0                              # excitatory synapse
                    forwardInputsE[post_ci] += wgt      #   - neuron ci spike's excitatory contribution to post_ci's synaptic current
                elseif wgt < 0                          # inhibitory synapse
                    forwardInputsI[post_ci] += wgt      #   - neuron ci spike's inhibitory contribution to post_ci's synaptic current
                end
            end #end loop over synaptic projections

            # (2) plastic connections
            # loop over neurons (indexed by j) postsynaptic to neuron ci. 
            # ncpOut[ci] is the number neurons postsynaptic neuron ci
            @static kind in [:train, :test, :train_test] && for j = 1:ncpOut[ci]
                post_ci = wpIndexOut[j,ci]                 # cell index of j_th postsynaptic neuron
                forwardInputsP[post_ci] += wpWeightOut[j,ci]    # neuron ci spike's contribution to post_ci's synaptic current
            end
        end
    end #end loop over neurons

    # save spiking activities produced at the current time step
    #   - forwardInputsPrev's will be used in the next time step to compute synaptic currents (xedecay, xidecay, xpdecay)
    #   - forwardSpikePrev will be used in the next time step to compute synapse-filter spikes (r)
    @static if p.K>0
        forwardInputsEPrev .= forwardInputsE
        forwardInputsIPrev .= forwardInputsI
    end
    @static kind in [:train, :test, :train_test] && (forwardInputsPPrev .= forwardInputsP)
    @static kind in [:train, :train_test] && (forwardSpikePrev .= forwardSpike) # if training, save spike trains

end #end loop over time

@static if kind == :init
    println("mean excitatory firing rate: ",mean(1000*ns[1:Ne]/train_time)," Hz")
    println("mean inhibitory firing rate: ",mean(1000*ns[(Ne+1):Ncells]/train_time)," Hz")
                
    ustd = mean(std(utmp, dims=1))
    return uavg, ns, ustd
end

@static if kind in [:test, :train_test]
    xtotal ./= xtotalcnt
    xebal ./= xebalcnt
    xibal ./= xibalcnt
    xplastic ./= xplasticcnt

    return ns, times, xtotal, xebal, xibal, xplastic, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell
end

end #end function
