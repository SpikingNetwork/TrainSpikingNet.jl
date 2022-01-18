function runtrain(dirData,p,w0Index,w0Weights,nc0, stim, xtarg,
    wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, 
    almOrd, matchedCells, ffwdRate)

# copy param
nloop = copy(p.nloop) # train param
penlambda = copy(p.penlambda)
penlamEE = copy(p.penlamEE)
penlamIE = copy(p.penlamIE)
penlamEI = copy(p.penlamEI)
penlamII = copy(p.penlamII)
penmu = copy(p.penmu)
frac = copy(p.fracTrained)
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

# set up variables
mu = zeros(Ncells)
mu[1:Ne] = (muemax-muemin)*rand(Ne) .+ muemin
mu[(Ne+1):Ncells] = (muimax-muimin)*rand(Ni) .+ muimin
thresh = zeros(Ncells)
thresh[1:Ne] .= threshe
thresh[(1+Ne):Ncells] .= threshi
tau = zeros(Ncells)
tau[1:Ne] .= taue
tau[(1+Ne):Ncells] .= taui
maxTimes = round(Int,maxrate*train_time/1000)
times = zeros(Ncells,maxTimes)
ns = zeros(Int,Ncells)
#####################################################
################ Feedforward inputs #################
#####################################################                    
times_ffwd = zeros(p.Lffwd, maxTimes)
ns_ffwd = zeros(Int, p.Lffwd)

forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
forwardInputsI = zeros(Ncells)
forwardInputsP = zeros(Ncells)
forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
forwardInputsIPrev = zeros(Ncells)
forwardInputsPPrev = zeros(Ncells)
forwardSpike = zeros(Ncells)
forwardSpikePrev = zeros(Ncells)
#####################################################
################ Feedforward inputs #################
#####################################################                    
ffwdSpike = zeros(p.Lffwd)
ffwdSpikePrev = zeros(p.Lffwd)

xedecay = zeros(Ncells)
xidecay = zeros(Ncells)
xpdecay = zeros(Ncells) 
synInputBalanced = zeros(Ncells)

v = rand(Ncells) #membrane voltage 

lastSpike = -100.0*ones(Ncells) #time of last spike
  
t = 0.0
r = zeros(Ncells)
#####################################################
################ Feedforward inputs #################
#####################################################                    
s = zeros(p.Lffwd)
bias = zeros(Ncells)
udrive_tmp = zeros(Ncells)

# set up correlation matrix

P = Vector{Array{Float64,2}}(); 
Px = Vector{Array{Int64,1}}();
#####################################################
################ Feedforward inputs #################
#####################################################                    
numFfwd = p.Lffwd
numExc = Int(p.Lexc)
numInh = Int(p.Linh)
numExcInh = numExc + numInh

# train a subset of excitatory neurons
for nid=1:length(matchedCells)    
    # index of a trained neuron
    ci = matchedCells[nid] # model neuron
    
    # neurons presynaptic to ci        
    push!(Px, wpIndexIn[ci,:])         

    # ----- Pinv: recurrent -----#
    # row sum penalty
    vec10 = [ones(numExc); zeros(numInh)];
    vec01 = [zeros(numExc); ones(numInh)];
    Pinv_rowsum = penmu*(vec10*vec10' + vec01*vec01')
    # L2-penalty
    Pinv_L2 = penlamEE*one(zeros(numExcInh,numExcInh))
    # Pinv: recurrent - L2 + Rowsum
    Pinv_rec = Pinv_L2 + Pinv_rowsum

    #####################################################
    ################ Feedforward inputs #################
    #####################################################                    
    # Include Pinv_ffwd to update ffwd weights
    # ----- Pinv: ffwd - L2 -----#
    Pinv_ffwd = p.penlamFF*one(zeros(numFfwd,numFfwd))

    #####################################################
    ################ Feedforward inputs #################
    #####################################################                    
    # Expand Pinv by numFfwd to include the ffwd weights
    # to augment recurrent and ffwd plastic inputs
    # ----- Pinv: total -----#
    Pinv = zeros(numExcInh+numFfwd, numExcInh+numFfwd)
    Pinv[1:numExcInh, 1:numExcInh] = Pinv_rec
    Pinv[numExcInh+1 : numExcInh+numFfwd, numExcInh+1 : numExcInh+numFfwd] = Pinv_ffwd

    push!(P, Pinv\one(zeros(numExcInh+numFfwd, numExcInh+numFfwd)))
end


for iloop =1:nloop
    println("Loop no. ",iloop) 

    start_time = time()

    for licki = 1:2

        # divide presynaptic neurons trained for lick left and right
        numExc = Int(p.Lexc)
        numInh = Int(p.Linh)
        numExcInh = numExc + numInh
        
        # initialize variables
        lastSpike .= -100.0
        ns .= 0
        ns_ffwd .= 0
        xedecay .= 0
        xidecay .= 0
        xpdecay .= 0
        r .= 0
        #####################################################
        ################ Feedforward inputs #################
        #####################################################                    
        s .= 0
        v = rand(Ncells)
        learn_seq = 1

        for ti=1:Nsteps
            t = dt*ti;
            forwardInputsE .= 0.0;
            forwardInputsI .= 0.0;
            forwardInputsP .= 0.0;
            forwardSpike .= 0.0;            
            #####################################################
            ################ Feedforward inputs #################
            #####################################################                    
            ffwdSpike .= 0.0;
            rndFfwd = rand(p.Lffwd)

            if t > Int(stim_off) && t <= Int(train_time) && mod(t, learn_every) == 0
                for nid = 1:length(matchedCells)
                    ci = matchedCells[nid] # model neuron
                    ci_alm = almOrd[nid] # alm neuron

                    #####################################################
                    ################ Feedforward inputs #################
                    #####################################################                    
                    # Augment rtrim and s since they both provide plastic inputs
                    # dim(raug) = numExcInh + numFfwd
                    rtrim = @view r[Px[nid]]                 
                    raug = [rtrim; s]

                    k = P[nid]*raug
                    vPv = raug'*k
                    den = 1.0/(1.0 + vPv[1])
                    BLAS.gemm!('N','T',-den,k,k,1.0,P[nid])

                    #####################################################
                    ################ Feedforward inputs #################
                    #####################################################                    
                    e  = wpWeightIn[ci,:]'*rtrim + wpWeightFfwd[licki][ci,:]'*s + synInputBalanced[ci] + mu[ci] - xtarg[licki][learn_seq,ci_alm]
                    dw = -e*k*den
                    wpWeightIn[ci,:] .+= dw[1 : numExcInh]
                    #####################################################
                    ################ Feedforward inputs #################
                    #####################################################
                    wpWeightFfwd[licki][ci,:] .+= dw[numExcInh+1 : end]
                end                
                wpWeightOut = convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
                learn_seq += 1
            end        

            for ci = 1:Ncells
                xedecay[ci] += -dt*xedecay[ci]/tauedecay + forwardInputsEPrev[ci]/tauedecay
                xidecay[ci] += -dt*xidecay[ci]/tauidecay + forwardInputsIPrev[ci]/tauidecay
                xpdecay[ci] += -dt*xpdecay[ci]/taudecay_plastic + forwardInputsPPrev[ci]/taudecay_plastic
                synInputBalanced[ci] = xedecay[ci] + xidecay[ci]
                synInput = synInputBalanced[ci] + xpdecay[ci]

                # if training, compute spike trains
                r[ci] += -dt*r[ci]/taudecay_plastic + forwardSpikePrev[ci]/taudecay_plastic

                # external input
                if t > Int(stim_on) && t < Int(stim_off) 
                    bias[ci] = mu[ci] + stim[licki][ti-Int(stim_on/dt),ci]
                else
                    bias[ci] = mu[ci]
                end

                #not in refractory period
                if t > (lastSpike[ci] + refrac)  
                    v[ci] += dt*((1/tau[ci])*(bias[ci]-v[ci] + synInput))
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

            #####################################################
            ################ Feedforward inputs #################
            #####################################################
            # s : filtered spike trains of ffwd spikes
            # ffwdRate : pre-defined spiking rate of external neurons
            # ffwdSpike
            # ffwdSpikePrev
            # (1) simulation: ffwd spikes are added to forwardInputsP
            # (2) training: ffwdSpikePrev computes the filtered ffwd spikes, s

            # External input to trained excitatory neurons
            if ti > Int(stim_off/dt)
                tidx = ti - Int(stim_off/dt)

                for ci = 1:p.Lffwd
                    # if training, filter the spikes
                    s[ci] += -dt*s[ci]/taudecay_plastic + ffwdSpikePrev[ci]/taudecay_plastic

                    # if Poisson neuron spiked
                    if rndFfwd[ci] < ffwdRate[licki][tidx,ci]/(1000/p.dt)
                        ffwdSpike[ci] = 1.
                        ns_ffwd[ci] = ns_ffwd[ci]+1
                        if ns_ffwd[ci] <= maxTimes
                            times_ffwd[ci,ns_ffwd[ci]] = t
                        end
                        for j = 1:Ncells
                            forwardInputsP[j] += wpWeightFfwd[licki][j,ci]
                        end #end loop over synaptic projections
                    end #end if spiked
                end #end loop over ffwd neurons
            end #end ffwd input

            forwardInputsEPrev = copy(forwardInputsE)
            forwardInputsIPrev = copy(forwardInputsI)
            forwardInputsPPrev = copy(forwardInputsP)
            forwardSpikePrev = copy(forwardSpike) # if training, compute spike trains
            #####################################################
            ################ Feedforward inputs #################
            #####################################################
            ffwdSpikePrev = copy(ffwdSpike) # if training, compute spike trains

        end #end loop over time
    end
    elapsed_time = time()-start_time
    println("elapsed time: ",elapsed_time)

end # end loop over trainings


return wpWeightIn, wpWeightOut, wpWeightFfwd

end
