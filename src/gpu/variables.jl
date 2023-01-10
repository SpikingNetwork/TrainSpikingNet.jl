mu = CuArray{p.FloatPrecision}(p.mu)  # external input

#synaptic time constants
invtauedecay = p.FloatPrecision(1/p.tauedecay)
invtauidecay = p.FloatPrecision(1/p.tauidecay)
if typeof(p.taudecay_plastic)<:Number
    invtaudecay_plastic = p.FloatPrecision(1/p.taudecay_plastic)
else
    invtaudecay_plastic = CuVector{p.FloatPrecision}(inv.(p.taudecay_plastic))
end

#spike thresholds
thresh = CuVector{p.FloatPrecision}(undef, p.Ncells)
thresh[1:p.Ne] .= p.threshe
thresh[(1+p.Ne):p.Ncells] .= p.threshi

#membrane time constants
tau = CuVector{p.FloatPrecision}(undef, p.Ncells)
tau[1:p.Ne] .= p.taue
tau[(1+p.Ne):p.Ncells] .= p.taui

maxTimes = round(Int, p.maxrate*p.train_time/1000)     # maximum number of spikes times to record
times = CuArray{Float64}(undef, p.Ncells, 1+maxTimes)  # times of recurrent spikes throughout trial
ns = CuVector{p.IntPrecision}(undef, p.Ncells)         # number of recurrent spikes in trial
times_ffwd = CuArray{Float64}(undef, p.Lffwd, 1+maxTimes)  # times of feed-forward spikes throughout trial
ns_ffwd = CuVector{p.IntPrecision}(undef, p.Lffwd)         # number of feed-forward spikes in trial

forwardInputsE = CuVector{p.FloatPrecision}(undef, p.Ncells+1)  # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = CuVector{p.FloatPrecision}(undef, p.Ncells+1)  # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = CuVector{p.FloatPrecision}(undef, p.Ncells+1)  # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = CuVector{p.FloatPrecision}(undef, p.Ncells+1) # copy of forwardInputsE from previous time step
forwardInputsIPrev = CuVector{p.FloatPrecision}(undef, p.Ncells+1) # copy of forwardInputsI from previous time step
forwardInputsPPrev = CuVector{p.FloatPrecision}(undef, p.Ncells+1) # copy of forwardInputsP from previous time step
forwardSpike = CuVector{p.FloatPrecision}(undef, p.Ncells) # spikes emitted by each recurrent neuron at one time step
forwardSpikePrev = CuVector{p.FloatPrecision}(undef, p.Ncells) # copy of forwardSpike from previous time step
ffwdSpike = CuVector{p.FloatPrecision}(undef, p.Lffwd) # spikes emitted by each feed-forward neuron at one time step
ffwdSpikePrev = CuVector{p.FloatPrecision}(undef, p.Lffwd) # copy of ffwdSpike from previous time step

xedecay = CuVector{p.FloatPrecision}(undef, p.Ncells)  # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = CuVector{p.FloatPrecision}(undef, p.Ncells)  # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = CuVector{p.FloatPrecision}(undef, p.Ncells)  # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = CuVector{p.FloatPrecision}(undef, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
synInput = CuVector{p.FloatPrecision}(undef, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = CuVector{p.FloatPrecision}(undef, p.Ncells)        # synapse-filtered recurrent spikes (i.e. filtered version of forwardSpike)
s = CuVector{p.FloatPrecision}(undef, p.Lffwd)         # synapse-filtered feed-forward spikes (i.e. filtered version of ffwdSpike)

bias = CuVector{p.FloatPrecision}(undef, p.Ncells)   # total external input to neurons
lastSpike = CuArray{Float64}(undef, p.Ncells)  # last time a neuron spiked

bnotrefrac = CuVector{Bool}(undef, p.Ncells)  # which recurrent neurons are not in the refractory period
bspike = CuVector{Bool}(undef, p.Ncells)   # which recurrent neurons spiked
plusone = p.FloatPrecision(1.0)
minusone = p.FloatPrecision(-1.0)
exactlyzero = p.FloatPrecision(0.0)
PScale = p.FloatPrecision(p.PScale)

vre = p.FloatPrecision(p.vre)  # reset voltage

raug = CuArray{p.FloatPrecision}(undef, p.Lexc+p.Linh+p.Lffwd, p.Ncells)
k = CuArray{p.FloatPrecision}(undef, p.Lexc+p.Linh+p.Lffwd, p.Ncells)
den = CuArray{p.FloatPrecision}(undef, p.Ncells)
e = CuArray{p.FloatPrecision}(undef, p.Ncells)
delta = CuArray{p.FloatPrecision}(undef, p.Lexc+p.Linh+p.Lffwd, p.Ncells)
v = CuVector{p.FloatPrecision}(undef, p.Ncells)  # membrane voltage
noise = CuArray{p.FloatPrecision}(undef, p.Ncells)  # actual noise added at current time step
sig = CUDA.fill(p.FloatPrecision(p.sig), p.Ncells)  # std dev of the Gaussian noise

rndFfwd = CuArray{p.FloatPrecision}(undef, p.Lffwd)  # uniform noise to generate Poisson feed-forward spikes
bspike_ffwd = CuVector{Bool}(undef, p.Lffwd)  # which feed-forward neurons spikes

learn_step = round(Int, p.learn_every/p.dt)
