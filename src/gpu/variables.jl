mu = CuArray{Param.FloatPrecision}(Param.mu)  # external input

#synaptic time constants
invtauedecay = Param.FloatPrecision(1/Param.tauedecay)
invtauidecay = Param.FloatPrecision(1/Param.tauidecay)
if typeof(Param.taudecay_plastic)<:Number
    invtaudecay_plastic = Param.FloatPrecision(1/Param.taudecay_plastic)
else
    invtaudecay_plastic = CuVector{Param.FloatPrecision}(inv.(Param.taudecay_plastic))
end

#spike thresholds
thresh = CuVector{Param.FloatPrecision}(undef, Param.Ncells)
thresh[1:Param.Ne] .= Param.threshe
thresh[(1+Param.Ne):Param.Ncells] .= Param.threshi

#membrane time constants
tau = CuVector{Param.FloatPrecision}(undef, Param.Ncells)
tau[1:Param.Ne] .= Param.taue
tau[(1+Param.Ne):Param.Ncells] .= Param.taui

maxTimes = round(Int, Param.maxrate * Param.train_time / 1000) # maximum number of spikes times to record
times = CuArray{Float64}(undef, Param.Ncells, 1+maxTimes)      # times of recurrent spikes throughout trial
ns = CuVector{Param.IntPrecision}(undef, Param.Ncells)         # number of recurrent spikes in trial
times_ffwd = CuArray{Float64}(undef, Param.Lffwd, 1+maxTimes)  # times of feed-forward spikes throughout trial
ns_ffwd = CuVector{Param.IntPrecision}(undef, Param.Lffwd)     # number of feed-forward spikes in trial

forwardInputsE = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)     # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)     # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)     # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1) # copy of forwardInputsE from previous time step
forwardInputsIPrev = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1) # copy of forwardInputsI from previous time step
forwardInputsPPrev = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1) # copy of forwardInputsP from previous time step
forwardSpike = CuVector{Param.FloatPrecision}(undef, Param.Ncells)         # spikes emitted by each recurrent neuron at one time step
forwardSpikePrev = CuVector{Param.FloatPrecision}(undef, Param.Ncells)     # copy of forwardSpike from previous time step
ffwdSpike = CuVector{Param.FloatPrecision}(undef, Param.Lffwd)             # spikes emitted by each feed-forward neuron at one time step
ffwdSpikePrev = CuVector{Param.FloatPrecision}(undef, Param.Lffwd)         # copy of ffwdSpike from previous time step

xedecay = CuVector{Param.FloatPrecision}(undef, Param.Ncells)          # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = CuVector{Param.FloatPrecision}(undef, Param.Ncells)          # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = CuVector{Param.FloatPrecision}(undef, Param.Ncells)          # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = CuVector{Param.FloatPrecision}(undef, Param.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
synInput = CuVector{Param.FloatPrecision}(undef, Param.Ncells)         # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = CuVector{Param.FloatPrecision}(undef, Param.Ncells)                # synapse-filtered recurrent spikes (i.e. filtered version of forwardSpike)
s = CuVector{Param.FloatPrecision}(undef, Param.Lffwd)                 # synapse-filtered feed-forward spikes (i.e. filtered version of ffwdSpike)

bias = CuVector{Param.FloatPrecision}(undef, Param.Ncells)   # total external input to neurons
lastSpike = CuArray{Float64}(undef, Param.Ncells)            # last time a neuron spiked

bnotrefrac = CuVector{Bool}(undef, Param.Ncells)  # which recurrent neurons are not in the refractory period
bspike = CuVector{Bool}(undef, Param.Ncells)      # which recurrent neurons spiked
plusone = Param.FloatPrecision(1.0)
minusone = Param.FloatPrecision(-1.0)
exactlyzero = Param.FloatPrecision(0.0)
PScale = Param.FloatPrecision(Param.PScale)

vre = Param.FloatPrecision(Param.vre)  # reset voltage

PLtot = Param.Lexc + Param.Linh + Param.Lffwd
raug = CuArray{Param.FloatPrecision}(undef, PLtot, Param.Ncells)
k = CuArray{Param.FloatPrecision}(undef, PLtot, Param.Ncells)
den = CuArray{Param.FloatPrecision}(undef, Param.Ncells)
e = CuArray{Param.FloatPrecision}(undef, Param.Ncells)
delta = CuArray{Param.FloatPrecision}(undef, PLtot, Param.Ncells)
v = CuVector{Param.FloatPrecision}(undef, Param.Ncells)         # membrane voltage
noise = CuArray{Param.FloatPrecision}(undef, Param.Ncells)      # actual noise added at current time step
sig = CUDA.fill(Param.FloatPrecision(Param.sig), Param.Ncells)  # std dev of the Gaussian noise

rndFfwd = CuArray{Param.FloatPrecision}(undef, Param.Lffwd)     # uniform noise to generate Poisson feed-forward spikes
bspike_ffwd = CuVector{Bool}(undef, Param.Lffwd)                # which feed-forward neurons spikes

learn_step = round(Int, Param.learn_every/Param.dt)
