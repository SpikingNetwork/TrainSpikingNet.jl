using Distributions

function runtest(p,w0Index,w0Weights,nc0,wpIndexOut,wpWeightOut,ncpOut,stim)

# copy param
nloop = copy(p.nloop) # train param
penlambda = copy(p.penlambda)
penmu = copy(p.penmu)
frac = copy(p.frac)
learn_every = copy(p.learn_every)
stim_on = copy(p.stim_on)
stim_off = copy(p.stim_off)
train_time = copy(p.train_time)
dt = copy(p.dt) # time param
Nsteps = copy(p.Nsteps) 
Ncells = copy(p.Ncells) # network param
Ne = copy(p.Ne)
Ni = copy(p.Ni)
taue = copy(p.taue) # neuron param
taui = copy(p.taui)
sqrtK = copy(p.sqrtK)
threshe = copy(p.threshe)
threshi = copy(p.threshi)
refrac = copy(p.refrac)
vre = copy(p.vre)
muemin = copy(p.muemin) # external input
muemax = copy(p.muemax)
muimin = copy(p.muimin)
muimax = copy(p.muimax)
tauedecay = copy(p.tauedecay) # synaptic time
tauidecay = copy(p.tauidecay)
taudecay_plastic = copy(p.taudecay_plastic)
maxrate = copy(p.maxrate)

invtauedecay = 1/tauedecay
invtauidecay = 1/tauidecay
invtaudecay_plastic = 1/taudecay_plastic

# set up variables
mu = zeros(Ncells)
mu[1:Ne] = (muemax-muemin)*rand(Ne) .+ muemin
mu[(Ne+1):Ncells] = (muimax-muimin)*rand(Ni) .+ muimin

thresh = zeros(Ncells)
thresh[1:Ne] .= threshe
thresh[(1+Ne):Ncells] .= threshi

invtau = zeros(Ncells)
invtau[1:Ne] .= 1/taue
invtau[(1+Ne):Ncells] .= 1/taui

maxTimes = round(Int,maxrate*train_time/1000)
times = zeros(Ncells,maxTimes)
ns = zeros(Int,Ncells)

forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
forwardInputsI = zeros(Ncells)
forwardInputsP = zeros(Ncells)
forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
forwardInputsIPrev = zeros(Ncells)
forwardInputsPPrev = zeros(Ncells)
forwardSpike = zeros(Ncells)
forwardSpikePrev = zeros(Ncells)

xedecay = zeros(Ncells)
xidecay = zeros(Ncells)
xpdecay = zeros(Ncells)
synInputBalanced = zeros(Ncells)

v = zeros(Ncells) #membrane voltage 

lastSpike = -100.0*ones(Ncells) #time of last spike
  
t = 0.0
bias = zeros(Ncells)

learn_nsteps = Int((p.train_time - p.stim_off)/p.learn_every)
learn_seq = 1
example_neurons = 25
wid = 50
widInc = Int(2*wid/p.learn_every - 1)

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
    forwardSpike .= 0.0;
    
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
        if t > Int(stim_on) && t < Int(stim_off)
            bias[ci] = mu[ci] + stim[ti-Int(stim_on/dt),ci]
        else
            bias[ci] = mu[ci]
        end

        #not in refractory period
        if t > (lastSpike[ci] + refrac)  
            v[ci] += dt*(invtau[ci]*(bias[ci]-v[ci] + synInput))
            if v[ci] > thresh[ci]  #spike occurred
                v[ci] = vre
                forwardSpike[ci] = 1.
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
    forwardSpikePrev = copy(forwardSpike) # if training, compute spike trains

end #end loop over time
print("\r")

# for k = 1:learn_nsteps
#     xtotal[k,:] = xtotal[k,:]/xtotalcnt[k]
#     xebal[k,:] = xebal[k,:]/xebalcnt[k]
#     xibal[k,:] = xibal[k,:]/xibalcnt[k]
#     xplastic[k,:] = xplastic[k,:]/xplasticcnt[k]
# end

return times, ns, vtotal_exccell, vtotal_inhcell, vebal_exccell, vibal_exccell, vebal_inhcell, vibal_inhcell, vplastic_exccell, vplastic_inhcell


end
