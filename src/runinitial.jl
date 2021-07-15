function runinitial(p,w0Index,w0Weights,nc0,stim)

# copy param
train_time = copy(p.train_time)
dt = copy(p.dt)
Nsteps = copy(p.Nsteps) # network param
Ncells = copy(p.Ncells)
Ne = copy(p.Ne)
Ni = copy(p.Ni)
taue = copy(p.taue) # neuron param
taui = copy(p.taui)
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
maxrate = copy(p.maxrate)

invtauedecay = 1/tauedecay
invtauidecay = 1/tauidecay

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
forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
forwardInputsIPrev = zeros(Ncells)

xedecay = zeros(Ncells)
xidecay = zeros(Ncells)

v = threshe*rand(Ncells) #membrane voltage 

lastSpike = -100.0*ones(Ncells) #time of last spike
  
uavg = zeros(Ncells)
utmp = zeros(Nsteps - Int(1000/p.dt),1000)
t = 0.0
r = zeros(Ncells)
bias = zeros(Ncells)

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

        if ti > Int(1000/p.dt) # 1000 ms
            uavg[ci] += synInput / (Nsteps - Int(1000/p.dt)) # save synInput
        end

        if ti > Int(1000/p.dt) && ci <=1000
            utmp[ti - Int(1000/p.dt), ci] = synInput
        end

        if !isempty(stim)
            if t > Int(stim_on) && t < Int(stim_off)
                bias[ci] = mu[ci] + stim[ti,ci]
            else
                bias[ci] = mu[ci]
            end
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
