X_bal = CuArray{Param.FloatPrecision}(Param.X_bal)  # external input

#synaptic time constants
invtau_bale = Param.FloatPrecision(1/Param.tau_bale)
invtau_bali = Param.FloatPrecision(1/Param.tau_bali)
if typeof(Param.tau_plas)<:Number
    invtau_plas = Param.FloatPrecision(1/Param.tau_plas)
else
    invtau_plas = CuVector{Param.FloatPrecision}(inv.(Param.tau_plas))
end

#spike thresholds
thresh = CuVector{Param.FloatPrecision}(undef, Param.Ncells)
thresh[1:Param.Ne] .= Param.threshe
thresh[(1+Param.Ne):Param.Ncells] .= Param.threshi

#membrane time constants
tau_mem = CuVector{Param.FloatPrecision}(undef, Param.Ncells)
tau_mem[1:Param.Ne] .= Param.tau_meme
tau_mem[(1+Param.Ne):Param.Ncells] .= Param.tau_memi

maxTimes = round(Int, Param.maxrate * Param.train_time / 1000)      # maximum number of spikes times to record
_times_precision = Dict(8=>UInt8, 16=>UInt16, 32=>UInt32, 64=>UInt64)[nextpow(2, log2(Param.Nsteps))]
times = CuArray{_times_precision}(undef, Param.Ncells, 1+maxTimes)  # times of recurrent spikes throughout trial
ns = CuVector{Param.IntPrecision}(undef, Param.Ncells)              # number of recurrent spikes in trial
timesX = CuArray{_times_precision}(undef, Param.LX, 1+maxTimes)     # times of feed-forward spikes throughout trial
nsX = CuVector{Param.IntPrecision}(undef, Param.LX)                 # number of feed-forward spikes in trial

inputsE = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)      # excitatory synaptic currents to neurons via balanced connections at one time step
inputsI = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)      # inhibitory synaptic currents to neurons via balanced connections at one time step
inputsP = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)      # synaptic currents to neurons via plastic connections at one time step
inputsEPrev = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)  # copy of inputsE from previous time step
inputsIPrev = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)  # copy of inputsI from previous time step
inputsPPrev = CuVector{Param.FloatPrecision}(undef, Param.Ncells+1)  # copy of inputsP from previous time step
spikes = CuVector{Param.FloatPrecision}(undef, Param.Ncells)         # spikes emitted by each recurrent neuron at one time step
spikesPrev = CuVector{Param.FloatPrecision}(undef, Param.Ncells)     # copy of spike from previous time step
spikesX = CuVector{Param.FloatPrecision}(undef, Param.LX)            # spikes emitted by each feed-forward neuron at one time step
spikesXPrev = CuVector{Param.FloatPrecision}(undef, Param.LX)        # copy of spikesX from previous time step

u_bale = CuVector{Param.FloatPrecision}(undef, Param.Ncells)   # synapse-filtered excitatory current (i.e. filtered version of inputsE)
u_bali = CuVector{Param.FloatPrecision}(undef, Param.Ncells)   # synapse-filtered inhibitory current (i.e. filtered version of inputsI)
uX_plas = CuVector{Param.FloatPrecision}(undef, Param.Ncells)  # synapse-filtered plastic current (i.e. filtered version of inputsP)
u_bal = CuVector{Param.FloatPrecision}(undef, Param.Ncells)    # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
u = CuVector{Param.FloatPrecision}(undef, Param.Ncells)        # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
r = CuVector{Param.FloatPrecision}(undef, Param.Ncells)        # synapse-filtered recurrent spikes (i.e. filtered version of spike)
rX = CuVector{Param.FloatPrecision}(undef, Param.LX)           # synapse-filtered feed-forward spikes (i.e. filtered version of spikesX)

X = CuVector{Param.FloatPrecision}(undef, Param.Ncells)   # total external input to neurons
lastSpike = CuArray{Float64}(undef, Param.Ncells)            # last time a neuron spiked

bnotrefrac = CuVector{Bool}(undef, Param.Ncells)  # which recurrent neurons are not in the refractory period
bspike = CuVector{Bool}(undef, Param.Ncells)      # which recurrent neurons spiked
plusone = Param.FloatPrecision(1.0)
minusone = Param.FloatPrecision(-1.0)
exactlyzero = Param.FloatPrecision(0.0)

vre = Param.FloatPrecision(Param.vre)  # reset voltage

PLtot = Param.Lexc + Param.Linh + Param.LX
raug = CuArray{Param.FloatPrecision}(undef, PLtot, Param.Ncells)
k = CuArray{Param.FloatPrecision}(undef, PLtot, Param.Ncells)
den = CuArray{Param.FloatPrecision}(undef, Param.Ncells)
e = CuArray{Param.FloatPrecision}(undef, Param.Ncells)
delta = CuArray{Param.FloatPrecision}(undef, PLtot, Param.Ncells)
v = CuVector{Param.FloatPrecision}(undef, Param.Ncells)         # membrane voltage
noise = CuArray{Param.FloatPrecision}(undef, Param.Ncells)      # actual noise added at current time step
sig = CUDA.fill(Param.FloatPrecision(Param.sig), Param.Ncells)  # std dev of the Gaussian noise

rndX = CuArray{Param.FloatPrecision}(undef, Param.LX)  # uniform noise to generate Poisson feed-forward spikes
bspikeX = CuVector{Bool}(undef, Param.LX)              # which feed-forward neurons spikes

learn_step = round(Int, Param.learn_every/Param.dt)
