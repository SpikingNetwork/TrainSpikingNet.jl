X_bal = Array{Param.FloatPrecision}(Param.X_bal)  # external input

#synaptic time constants
invtau_bale = Param.FloatPrecision(1/Param.tau_bale)
invtau_bali = Param.FloatPrecision(1/Param.tau_bali)
if typeof(Param.tau_plas)<:Number
    invtau_plas = Param.FloatPrecision(1/Param.tau_plas)
else
    invtau_plas = Vector{Param.FloatPrecision}(inv.(Param.tau_plas))
end

#spike thresholds
thresh = Vector{Param.FloatPrecision}(undef, Param.Ncells)
thresh[1:Param.Ne] .= Param.threshe
thresh[(1+Param.Ne):Param.Ncells] .= Param.threshi

#membrane time constants
tau_mem = Vector{Param.FloatPrecision}(undef, Param.Ncells)
tau_mem[1:Param.Ne] .= Param.tau_meme
tau_mem[(1+Param.Ne):Param.Ncells] .= Param.tau_memi

maxTimes = round(Int, Param.maxrate * Param.train_time / 1000)  # maximum number of spikes times to record
_times_precision = Dict(8=>UInt8, 16=>UInt16, 32=>UInt32, 64=>UInt64)[nextpow(2, log2(Param.Nsteps))]
times = Array{_times_precision}(undef, Param.Ncells, maxTimes)  # times of recurrent spikes throughout trial
ns = Vector{Param.IntPrecision}(undef, Param.Ncells)            # number of recurrent spikes in trial
timesX = Array{_times_precision}(undef, Param.LX, maxTimes)     # times of feed-forward spikes throughout trial
nsX = Vector{Param.IntPrecision}(undef, Param.LX)               # number of feed-forward spikes in trial

inputsE = Vector{Param.FloatPrecision}(undef, Param.Ncells)      # excitatory synaptic currents to neurons via balanced connections at one time step
inputsI = Vector{Param.FloatPrecision}(undef, Param.Ncells)      # inhibitory synaptic currents to neurons via balanced connections at one time step
inputsP = Vector{Param.FloatPrecision}(undef, Param.Ncells)      # synaptic currents to neurons via plastic connections at one time step
inputsEPrev = Vector{Param.FloatPrecision}(undef, Param.Ncells)  # copy of inputsE from previous time step
inputsIPrev = Vector{Param.FloatPrecision}(undef, Param.Ncells)  # copy of inputsI from previous time step
inputsPPrev = Vector{Param.FloatPrecision}(undef, Param.Ncells)  # copy of inputsP from previous time step
spikes = Vector{Param.FloatPrecision}(undef, Param.Ncells)       # spikes emitted by each recurrent neuron at one time step
spikesPrev = Vector{Param.FloatPrecision}(undef, Param.Ncells)   # copy of spike from previous time step
spikesX = Vector{Param.FloatPrecision}(undef, Param.LX)          # spikes emitted by each feed-forward neuron at one time step
spikesXPrev = Vector{Param.FloatPrecision}(undef, Param.LX)      # copy of spikesX from previous time step

u_bale = Vector{Param.FloatPrecision}(undef, Param.Ncells)   # synapse-filtered excitatory current (i.e. filtered version of inputsE)
u_bali = Vector{Param.FloatPrecision}(undef, Param.Ncells)   # synapse-filtered inhibitory current (i.e. filtered version of inputsI)
uX_plas = Vector{Param.FloatPrecision}(undef, Param.Ncells)  # synapse-filtered plastic current (i.e. filtered version of inputsP)
u_bal = Vector{Param.FloatPrecision}(undef, Param.Ncells)    # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
u = Vector{Param.FloatPrecision}(undef, Param.Ncells)        # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
r = Vector{Param.FloatPrecision}(undef, Param.Ncells)        # synapse-filtered recurrent spikes (i.e. filtered version of spike)
rX = Vector{Param.FloatPrecision}(undef, Param.LX)           # synapse-filtered feed-forward spikes (i.e. filtered version of spikesX)

X = Vector{Param.FloatPrecision}(undef, Param.Ncells)  # total external input to neurons
lastSpike = Array{Float64}(undef, Param.Ncells)           # last time a neuron spiked

plusone = Param.FloatPrecision(1.0)
exactlyzero = Param.FloatPrecision(0.0)

vre = Param.FloatPrecision(Param.vre)  # reset voltage

uavg = zeros(Float64, Param.Ncells)  # average synaptic input
utmp = Matrix{Float64}(undef, Param.Nsteps - round(Int, 1000/Param.dt), 1000)

PLtot = Param.Lexc + Param.Linh + Param.LX
raug = Matrix{Param.FloatPrecision}(undef, PLtot, Threads.nthreads())
k = Matrix{Param.FloatPrecision}(undef, PLtot, Threads.nthreads())
delta = Matrix{Param.FloatPrecision}(undef, PLtot, Threads.nthreads())
v = Vector{Param.FloatPrecision}(undef, Param.Ncells)      # membrane voltage
noise = Vector{Param.FloatPrecision}(undef, Param.Ncells)  # actual noise added at current time step
sig = fill(Param.FloatPrecision(Param.sig), Param.Ncells)  # std dev of the Gaussian noise

rndX = Vector{Param.FloatPrecision}(undef, Param.LX)    # uniform noise to generate Poisson feed-forward spikes

learn_step = round(Int, Param.learn_every/Param.dt)
