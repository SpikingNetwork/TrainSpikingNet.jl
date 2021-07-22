@eval function $(Symbol("loop_",kind))(learn_every, stim_on, stim_off,
    train_time, dt, Nsteps, Ncells, Ne, refrac, vre, invtauedecay,
    invtauidecay, invtaudecay_plastic, mu, thresh, invtau, ns, forwardInputsE,
    forwardInputsI, forwardInputsP, forwardInputsEPrev, forwardInputsIPrev,
    forwardInputsPPrev, forwardSpike, forwardSpikePrev, xedecay, xidecay,
    xpdecay, synInputBalanced, r, bias, example_neurons, lastSpike, plusone,
    minusone, k, v, P, Px, w0Index, w0Weights, nc0, stim, xtarg, wpIndexIn,
    wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut,
    uavg, utmp)

@static if kind == :test
    learn_nsteps = Int((train_time - stim_off)/learn_every)
    wid = 50   
    widInc = Int(2*wid/learn_every - 1)
            
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
    xtotalcnt = zeros(learn_nsteps)
    xebalcnt = zeros(learn_nsteps)
    xibalcnt = zeros(learn_nsteps)
    xplasticcnt = zeros(learn_nsteps)
end

@static kind == :train && (learn_seq = 1)

# start the actual training
for ti=1:Nsteps
    t = dt*ti;

    # reset spiking activities from the previous time step
    forwardInputsE .= 0.0;
    forwardInputsI .= 0.0;
    @static kind in [:train,:test] && (forwardInputsP .= 0.0;)
    @static kind==:train && (forwardSpike .= 0.0;)

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

    @static kind==:train && if t > stim_off && t <= train_time && mod(t, learn_every) == 0
        wpWeightIn, wpWeightOut, learn_seq = rls(k, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone)
    end

    @static if kind == :test
        if t > stim_off && t <= train_time && mod(t,1.0) == 0
            startInd = floor(Int, (t - stim_off - wid)/learn_every + 1)
            endInd = min(startInd + widInc, learn_nsteps)
            startInd = max(startInd, 1)
            xtotalcnt[startInd:endInd] .+= 1
            xebalcnt[startInd:endInd] .+= 1
            xibalcnt[startInd:endInd] .+= 1
            xplasticcnt[startInd:endInd] .+= 1
        end
    end

    # update network activities:
    #   - synaptic currents (xedecay, xidecay, xpdecay)
    #   - membrane potential (v) 
    #   - weighted spikes received by each neuron (forwardInputsE, forwardInputsI, forwardInputsP)
    #   - activity variables used for training
    #       * spikes emitted by each neuron (forwardSpike)
    #       * synapse-filtered spikes emitted by each neuron (r)        
    for ci = 1:Ncells
        xedecay[ci] += -dt*xedecay[ci]*invtauedecay + forwardInputsEPrev[ci]*invtauedecay
        xidecay[ci] += -dt*xidecay[ci]*invtauidecay + forwardInputsIPrev[ci]*invtauidecay
        @static if kind in [:train, :test]
            xpdecay[ci] += -dt*xpdecay[ci]*invtaudecay_plastic + forwardInputsPPrev[ci]*invtaudecay_plastic
            synInputBalanced[ci] = xedecay[ci] + xidecay[ci]
            synInput = synInputBalanced[ci] + xpdecay[ci]
        end
        @static if kind == :init
            synInput = xedecay[ci] + xidecay[ci]
        end

        @static if kind == :init
            if ti > Int(1000/dt) # 1000 ms
                uavg[ci] += synInput / (Nsteps - round(Int,1000/dt)) # save synInput
            end

            if ti > Int(1000/dt) && ci <=1000
                utmp[ti - round(Int,1000/dt), ci] = synInput
            end
        end

        @static if kind == :test
            # saved for visualization
            if ci <= example_neurons
                vtotal_exccell[ti,ci] = synInput
                vebal_exccell[ti,ci] = xedecay[ci]
                vibal_exccell[ti,ci] = xidecay[ci]
                vplastic_exccell[ti,ci] = xpdecay[ci]
            elseif ci >= Ncells - example_neurons + 1
                vtotal_inhcell[ti,ci-Ncells+example_neurons] = synInput
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
        @static if kind == :train
            r[ci] += -dt*r[ci]*invtaudecay_plastic + forwardSpikePrev[ci]*invtaudecay_plastic
        end

        # external inputs
        #   - mu: default inputs to maintain the balanced state
        #   - stim: inputs that trigger the learned responses
        #         : applied within the time interval [stim_on, stim_off]
        @static if kind in [:train, :test]
            if t > stim_on && t < stim_off
                bias[ci] = mu[ci] + stim[ti-round(Int,stim_on/dt),ci]
            else
                bias[ci] = mu[ci]
            end
        end
        @static if kind == :init
            bias[ci] = mu[ci]
        end

        # neuron ci not in refractory period
        if t > (lastSpike[ci] + refrac)  
            # update membrane potential
            v[ci] += dt*(invtau[ci]*(bias[ci]-v[ci] + synInput))

            #spike occurred
            if v[ci] > thresh[ci]                      
                v[ci] = vre                 # reset voltage
                @static kind==:train && (forwardSpike[ci] = 1.)   # record that neuron ci spiked. Used for computing r[ci]
                lastSpike[ci] = t           # record neuron ci's last spike time. Used for checking ci is not in refractory period
                ns[ci] = ns[ci]+1           # number of spikes neuron ci emitted

                # Accumulate the contribution of spikes to postsynaptic currents
                # Network connectivity is divided into two parts:
                #   - balanced connections (static) 
                #   - plastic connections

                # (1) balanced connections (static)
                # loop over neurons (indexed by j) postsynaptic to neuron ci.                     
                # nc0[ci] is the number neurons postsynaptic neuron ci
                for j = 1:nc0[ci]                       
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
                @static kind in [:train,:test] && for j = 1:ncpOut[ci]
                    post_ci = Int(wpIndexOut[j,ci])                 # cell index of j_th postsynaptic neuron
                    forwardInputsP[post_ci] += wpWeightOut[j,ci]    # neuron ci spike's contribution to post_ci's synaptic current
                end
            end #end if(spike occurred)
        end #end not in refractory period
    end #end loop over neurons

    # save spiking activities produced at the current time step
    #   - forwardInputsPrev's will be used in the next time step to compute synaptic currents (xedecay, xidecay, xpdecay)
    #   - forwardSpikePrev will be used in the next time step to compute synapse-filter spikes (r)
    forwardInputsEPrev = copy(forwardInputsE)
    forwardInputsIPrev = copy(forwardInputsI)
    @static kind in [:train,:test] && (forwardInputsPPrev = copy(forwardInputsP))
    @static kind==:train && (forwardSpikePrev = copy(forwardSpike)) # if training, save spike trains

end #end loop over time

@static if kind == :init
    println("mean excitatory firing rate: ",mean(1000*ns[1:Ne]/train_time)," Hz")
    println("mean inhibitory firing rate: ",mean(1000*ns[(Ne+1):Ncells]/train_time)," Hz")
                
    ustd = mean(std(utmp, dims=1))
    return uavg, ns, ustd
end

@static if kind == :test
    xtotal ./ xtotalcnt
    xebal ./ xebalcnt
    xibal ./ xibalcnt
    xplastic ./ xplasticcnt
 
    return xtotal, xebal, xibal, xplastic, ns, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell
end

end #end function
