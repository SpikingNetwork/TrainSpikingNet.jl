using Distributions

function runtest(stim_on, stim_off, dt, Nsteps, Ncells, refrac, vre,
    invtauedecay, invtauidecay, invtaudecay_plastic, mu, thresh, invtau,
    maxTimes, times, ns, forwardInputsE, forwardInputsI, forwardInputsP,
    forwardInputsEPrev, forwardInputsIPrev, forwardInputsPPrev, xedecay,
    xidecay, xpdecay, synInputBalanced, v, lastSpike, bias, example_neurons,
    w0Index, w0Weights, nc0, wpIndexOut, wpWeightOut, ncpOut, stim)

#learn_nsteps = Int((p.train_time - p.stim_off)/p.learn_every)
#learn_seq = 1
#wid = 50
#widInc = Int(2*wid/p.learn_every - 1)

vtotal_exccell = zeros(Nsteps,example_neurons)
vtotal_inhcell = zeros(Nsteps,example_neurons)
vebal_exccell = zeros(Nsteps,example_neurons)
vibal_exccell = zeros(Nsteps,example_neurons)
vebal_inhcell = zeros(Nsteps,example_neurons)
vibal_inhcell = zeros(Nsteps,example_neurons)
vplastic_exccell = zeros(Nsteps,example_neurons)
vplastic_inhcell = zeros(Nsteps,example_neurons)
# xtotal = zeros(learn_nsteps,Ncells)
# xebal = zeros(learn_nsteps,Ncells)
# xibal = zeros(learn_nsteps,Ncells)
# xplastic = zeros(learn_nsteps,Ncells)
# xtotalcnt = zeros(learn_nsteps)
# xebalcnt = zeros(learn_nsteps)
# xibalcnt = zeros(learn_nsteps)
# xplasticcnt = zeros(learn_nsteps)

for ti=1:Nsteps
    if mod(ti,Nsteps/100) == 1  #print percent complete
        print("\r",round(Int,100*ti/Nsteps))
    end

    t = dt*ti;
    forwardInputsE .= 0.0;
    forwardInputsI .= 0.0;
    forwardInputsP .= 0.0;
    
    for ci = 1:Ncells
        xedecay[ci] += -dt*xedecay[ci]*invtauedecay + forwardInputsEPrev[ci]*invtauedecay
        xidecay[ci] += -dt*xidecay[ci]*invtauidecay + forwardInputsIPrev[ci]*invtauidecay
        xpdecay[ci] += -dt*xpdecay[ci]*invtaudecay_plastic + forwardInputsPPrev[ci]*invtaudecay_plastic
        synInputBalanced[ci] = xedecay[ci] + xidecay[ci]
        synInput = synInputBalanced[ci] + xpdecay[ci]

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

        # # save rolling average for analysis
        # if t > Int(stim_off) && t <= Int(train_time) && mod(t,1.0) == 0
        #     xtotal[:,ci], xtotalcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xtotal[:,ci],xtotalcnt,synInput,ci)
        #     xebal[:,ci], xebalcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xebal[:,ci],xebalcnt,xedecay[ci],ci)
        #     xibal[:,ci], xibalcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xibal[:,ci],xibalcnt,xidecay[ci],ci)
        #     xplastic[:,ci], xplasticcnt = funRollingAvg(p,t,wid,widInc,learn_nsteps,xplastic[:,ci],xplasticcnt,xpdecay[ci],ci)
        # end
        
        # external input
        if t > stim_on && t < stim_off
            bias[ci] = mu[ci] + stim[ti-round(Int,stim_on/dt),ci]
        else
            bias[ci] = mu[ci]
        end

        #not in refractory period
        if t > (lastSpike[ci] + refrac)  
            v[ci] += dt*(invtau[ci]*(bias[ci]-v[ci] + synInput))
            if v[ci] > thresh[ci]  #spike occurred
                v[ci] = vre
                lastSpike[ci] = t
                ns[ci] = ns[ci]+1
                if ns[ci] <= maxTimes
                    times[ci,ns[ci]] = t
                end
                for j = 1:nc0[ci]
                    wgt = w0Weights[j,ci]
                    cell = w0Index[j,ci]
                    if wgt > 0  #E synapse
                        forwardInputsE[cell] += wgt
                    elseif wgt < 0  #I synapse
                        forwardInputsI[cell] += wgt
                    end
                end #end loop over synaptic projections
                for j = 1:ncpOut[ci]
                    cell = Int(wpIndexOut[j,ci])
                    forwardInputsP[cell] += wpWeightOut[j,ci]
                end
            end #end if(spike occurred)
        end #end not in refractory period
    end #end loop over neurons

    forwardInputsEPrev = copy(forwardInputsE)
    forwardInputsIPrev = copy(forwardInputsI)
    forwardInputsPPrev = copy(forwardInputsP)

end #end loop over time
print("\r")

# for k = 1:learn_nsteps
#     xtotal[k,:] = xtotal[k,:]/xtotalcnt[k]
#     xebal[k,:] = xebal[k,:]/xebalcnt[k]
#     xibal[k,:] = xibal[k,:]/xibalcnt[k]
#     xplastic[k,:] = xplastic[k,:]/xplasticcnt[k]
# end

return vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell

end
