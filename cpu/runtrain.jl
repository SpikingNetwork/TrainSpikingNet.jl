function runtrain(p,w0Index,w0Weights,nc0,stim,xtarg,wpIndexIn,wpIndexOut,wpIndexConvert,wpWeightIn,wpWeightOut,ncpIn,ncpOut)

# copy simulation param
nloop = copy(p.nloop)                       # number of training iterations
penlambda = copy(p.penlambda)               # L2-penalty
penlamEE = copy(p.penlamEE)                 # not used
penlamIE = copy(p.penlamIE)                 # not used
penlamEI = copy(p.penlamEI)                 # not used
penlamII = copy(p.penlamII)                 # not used
penmu = copy(p.penmu)                       # Rowsum-penalty
frac = copy(p.frac)                         # not used
learn_every = copy(p.learn_every)           # recursive least squares algorithm updates the plastic weights every learn_every (=10ms)
stim_on = copy(p.stim_on)                   # time at which the stimulus triggering the learned response is turned on (800ms) 
stim_off = copy(p.stim_off)                 # time at which the stimulus triggering the learned response is turned off (1000ms)
train_time = copy(p.train_time)             # total training time (2000ms)
dt = copy(p.dt)                             # simulation time step (0.1ms)
Nsteps = copy(p.Nsteps)                     # number of simulation time steps
Ncells = copy(p.Ncells)                     # number of cells
Ne = copy(p.Ne)                             # number of excitatory cells
Ni = copy(p.Ni)                             # number of inhibitory cells
taue = copy(p.taue)                         # membrane time constant of excitatory cells (10ms)
taui = copy(p.taui)                         # membrane time constant of inhibitory cells (10ms)
sqrtK = copy(p.sqrtK)                       # sqrt(K) where K is the average number of exc/inh synaptic connections to a neuron
threshe = copy(p.threshe)                   # spike threshold of excitatory cells (1)
threshi = copy(p.threshi)                   # spike threshold of inhibitory cells (1)
refrac = copy(p.refrac)                     # refractory period (0, no refractory period)
vre = copy(p.vre)                           # voltage reset after spike (0)
muemin = copy(p.muemin)                     # external input to excitatory neurons (min)
muemax = copy(p.muemax)                     # external input to excitatory neurons (max)
muimin = copy(p.muimin)                     # external input to inhibitory neurons (min)
muimax = copy(p.muimax)                     # external input to inhibitory neurons (max)
tauedecay = copy(p.tauedecay)               # excitatory synaptic decay time constant (3ms) - for balanced connectivity (static)
tauidecay = copy(p.tauidecay)               # inhibitory synaptic decay time constant (3ms) - for balanced connectivity (static)
taudecay_plastic = copy(p.taudecay_plastic) # synaptic decay time constant (150ms) - for plastic connectivity 
maxrate = copy(p.maxrate)                   # maximum firing rate allowed (500Hz)

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

forwardInputsE = zeros(Ncells)              # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = zeros(Ncells)              # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = zeros(Ncells)              # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = zeros(Ncells)          # copy of forwardInputsE from previous time step
forwardInputsIPrev = zeros(Ncells)          # copy of forwardInputsI from previous time step
forwardInputsPPrev = zeros(Ncells)          # copy of forwardInputsP from previous time step
forwardSpike = zeros(Ncells)                # spikes emitted by each neuron at one time step
forwardSpikePrev = zeros(Ncells)            # copy of forwardSpike from previous time step

xedecay = zeros(Ncells)                     # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = zeros(Ncells)                     # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = zeros(Ncells)                     # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = zeros(Ncells)            # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = zeros(Ncells)                           # synapse-filtered spikes (i.e. filtered version of forwardSpike)

bias = zeros(Ncells)                        # total external input to neurons
lastSpike = -100.0*ones(Ncells)             # last time a neuron spiked

k = Vector{Float64}(undef, 2*L)

# start training loops
for iloop =1:nloop
    println("Loop no. ",iloop) 

    # initialize variables
    lastSpike .= -100.0
    ns .= 0
    xedecay .= 0
    xidecay .= 0
    xpdecay .= 0
    r .= 0
    v = rand(Ncells) # membrane potentials have random initial values
    learn_seq = 1

    start_time = time()

    # start the actual training
    for ti=1:Nsteps
        t = dt*ti;

        # reset spiking activities from the previous time step
        forwardInputsE .= 0.0;
        forwardInputsI .= 0.0;
        forwardInputsP .= 0.0;
        forwardSpike .= 0.0;

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
        #             - Ncell x Kin matrix where Kin = p.Lexc + p.Linh is the number of incoming plastic synapses to each neuron
        #             - ith row, wpWeightIn[i,:]:
        #                 - weights of the incoming connections to neuron i
        #                 - each row of wpWeightIn will be updated independently by the rls algorithm. (see line 153)
        # wpIndexIn:  - Ncell x Kin matrix where Kin = p.Lexc + p.Linh
        #             - ith row, wpIndexIn[i,:]:
        #                 - Indices of presynaptic neurons that connect to neuron i
        #                 - Fixed throughout the simulation. Used to define Px (see line 83)
        # wpWeightOut: - plastic weights used for simulating network activities
        #              - Kout x Ncell matrix where Kout is the number of outgoing plastic synapses from each neuron
        #              - the actual number of outgoing plastic synapses is different across neurons, so we chose a fixed number Kout >= p.Lexc + p.Linh
        #              - ith column, wpWeightOut[:,i]:
        #                  - weights of the outgoing connections from neuron i
        #                  - Used to compute forwardInputsP (see line 221)
        # wpIndexOut:  - Kout x Ncell matrix
        #              - ith column, wpIndexOut[:,i]:
        #                  - Indices of postsynaptic neurons that neuron i connect to
        #                  - Fixed throughout the simulation. Used to compute forwardInputsP (see line 221)

        if t > Int(stim_off) && t <= Int(train_time) && mod(t, learn_every) == 0
            wpWeightIn, wpWeightOut, learn_seq = rls(k, p, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut)
        end

        # update network activities:
        #   - synaptic currents (xedecay, xidecay, xpdecay)
        #   - membrane potential (v) 
        #   - weighted spikes received by each neuron (forwardInputsE, forwardInputsI, forwardInputsP)
        #   - activity variables used for training
        #       * spikes emitted by each neuron (forwardSpike)
        #       * synapse-filtered spikes emitted by each neuron (r)        
        for ci = 1:Ncells
            xedecay[ci] += -dt*xedecay[ci]/tauedecay + forwardInputsEPrev[ci]/tauedecay
            xidecay[ci] += -dt*xidecay[ci]/tauidecay + forwardInputsIPrev[ci]/tauidecay
            xpdecay[ci] += -dt*xpdecay[ci]/taudecay_plastic + forwardInputsPPrev[ci]/taudecay_plastic
            synInputBalanced[ci] = xedecay[ci] + xidecay[ci]
            synInput = synInputBalanced[ci] + xpdecay[ci]

            # if training, compute synapse-filtered spike trains
            r[ci] += -dt*r[ci]/taudecay_plastic + forwardSpikePrev[ci]/taudecay_plastic

            # external inputs
            #   - mu: default inputs to maintain the balanced state
            #   - stim: inputs that trigger the learned responses
            #         : applied within the time interval [stim_on, stim_off]
            if t > Int(stim_on) && t < Int(stim_off)
                bias[ci] = mu[ci] + stim[ti-round(Int,stim_on/dt),ci]
            else
                bias[ci] = mu[ci]
            end

            # neuron ci not in refractory period
            if t > (lastSpike[ci] + refrac)  
                # update membrane potential
                v[ci] += dt*((1/tau[ci])*(bias[ci]-v[ci] + synInput))

                #spike occurred
                if v[ci] > thresh[ci]                      
                    v[ci] = vre                 # reset voltage
                    forwardSpike[ci] = 1.       # record that neuron ci spiked. Used for computing r[ci]
                    lastSpike[ci] = t           # record neuron ci's last spike time. Used for checking ci is not in refractory period
                    ns[ci] = ns[ci]+1           # number of spikes neuron ci emitted
                    if ns[ci] <= maxTimes       # maxTimes is the maximum number of spikes we will track (500Hz)
                        times[ci,ns[ci]] = t    # record spike times
                    end

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
                    for j = 1:ncpOut[ci]
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
        forwardInputsPPrev = copy(forwardInputsP)
        forwardSpikePrev = copy(forwardSpike) # if training, save spike trains

    end #end loop over time
elapsed_time = time()-start_time
println("elapsed time: ",elapsed_time)

end # end loop over trainings


return wpWeightIn, wpWeightOut

end
