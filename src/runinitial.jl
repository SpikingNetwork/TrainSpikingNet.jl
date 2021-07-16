function runinitial(train_time,dt,Nsteps,Ncells,Ne,refrac,vre,invtauedecay,invtauidecay,mu,thresh,invtau,maxTimes,times,ns,forwardInputsE,forwardInputsI,forwardInputsEPrev,forwardInputsIPrev,xedecay,xidecay,v,lastSpike,uavg,utmp,bias,w0Index,w0Weights,nc0)

for ti=1:Nsteps
    if mod(ti,Nsteps/100) == 1  #print percent complete
        print("\r",round(Int,100*ti/Nsteps))
    end
    t = dt*ti;
    forwardInputsE .= 0.0;
    forwardInputsI .= 0.0;
    for ci = 1:Ncells
        xedecay[ci] += -dt*xedecay[ci]*invtauedecay + forwardInputsEPrev[ci]*invtauedecay
        xidecay[ci] += -dt*xidecay[ci]*invtauidecay + forwardInputsIPrev[ci]*invtauidecay
        synInput = xedecay[ci] + xidecay[ci]

        if ti > Int(1000/dt) # 1000 ms
            uavg[ci] += synInput / (Nsteps - round(Int,1000/dt)) # save synInput
        end

        if ti > Int(1000/dt) && ci <=1000
            utmp[ti - round(Int,1000/dt), ci] = synInput
        end

        bias[ci] = mu[ci]

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
                    if w0Weights[j,ci] > 0  #E synapse
                        forwardInputsE[w0Index[j,ci]] += w0Weights[j,ci]
                    elseif w0Weights[j,ci] < 0  #I synapse
                        forwardInputsI[w0Index[j,ci]] += w0Weights[j,ci]
                    end
                end #end loop over synaptic projections
            end #end if(spike occurred)
        end #end not in refractory period
    end #end loop over neurons

    forwardInputsEPrev = copy(forwardInputsE)
    forwardInputsIPrev = copy(forwardInputsI)

end #end loop over time
print("\r")
println("mean excitatory firing rate: ",mean(1000*ns[1:Ne]/train_time)," Hz")
println("mean inhibitory firing rate: ",mean(1000*ns[(Ne+1):Ncells]/train_time)," Hz")

ustd = mean(std(utmp, dims=1))
return uavg, ns, ustd

end
