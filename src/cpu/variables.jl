mu = Array{Param.FloatPrecision}(Param.mu)  # external input

#synaptic time constants
invtauedecay = Param.FloatPrecision(1/Param.tauedecay)
invtauidecay = Param.FloatPrecision(1/Param.tauidecay)
if typeof(Param.taudecay_plastic)<:Number
    invtaudecay_plastic = Param.FloatPrecision(1/Param.taudecay_plastic)
else
    invtaudecay_plastic = Vector{Param.FloatPrecision}(inv.(Param.taudecay_plastic))
end

#spike thresholds
thresh = Vector{Param.FloatPrecision}(undef, Param.Ncells)
thresh[1:Param.Ne] .= Param.threshe
thresh[(1+Param.Ne):Param.Ncells] .= Param.threshi

#membrane time constants
tau = Vector{Param.FloatPrecision}(undef, Param.Ncells)
tau[1:Param.Ne] .= Param.taue
tau[(1+Param.Ne):Param.Ncells] .= Param.taui

maxTimes = round(Int, Param.maxrate * Param.train_time / 1000)  # maximum number of spikes times to record
times = Array{Float64}(undef, Param.Ncells, maxTimes)           # times of recurrent spikes throughout trial
ns = Vector{Param.IntPrecision}(undef, Param.Ncells)            # number of recurrent spikes in trial
times_ffwd = Array{Float64}(undef, Param.Lffwd, maxTimes)       # times of feed-forward spikes throughout trial
ns_ffwd = Vector{Param.IntPrecision}(undef, Param.Lffwd)        # number of feed-forward spikes in trial

forwardInputsE = Vector{Param.FloatPrecision}(undef, Param.Ncells)     # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = Vector{Param.FloatPrecision}(undef, Param.Ncells)     # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = Vector{Param.FloatPrecision}(undef, Param.Ncells)     # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = Vector{Param.FloatPrecision}(undef, Param.Ncells) # copy of forwardInputsE from previous time step
forwardInputsIPrev = Vector{Param.FloatPrecision}(undef, Param.Ncells) # copy of forwardInputsI from previous time step
forwardInputsPPrev = Vector{Param.FloatPrecision}(undef, Param.Ncells) # copy of forwardInputsP from previous time step
forwardSpike = Vector{Param.FloatPrecision}(undef, Param.Ncells)       # spikes emitted by each recurrent neuron at one time step
forwardSpikePrev = Vector{Param.FloatPrecision}(undef, Param.Ncells)   # copy of forwardSpike from previous time step
ffwdSpike = Vector{Param.FloatPrecision}(undef, Param.Lffwd)           # spikes emitted by each feed-forward neuron at one time step
ffwdSpikePrev = Vector{Param.FloatPrecision}(undef, Param.Lffwd)       # copy of ffwdSpike from previous time step

xedecay = Vector{Param.FloatPrecision}(undef, Param.Ncells)          # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = Vector{Param.FloatPrecision}(undef, Param.Ncells)          # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = Vector{Param.FloatPrecision}(undef, Param.Ncells)          # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = Vector{Param.FloatPrecision}(undef, Param.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
synInput = Vector{Param.FloatPrecision}(undef, Param.Ncells)         # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = Vector{Param.FloatPrecision}(undef, Param.Ncells)                # synapse-filtered recurrent spikes (i.e. filtered version of forwardSpike)
s = Vector{Param.FloatPrecision}(undef, Param.Lffwd)                 # synapse-filtered feed-forward spikes (i.e. filtered version of ffwdSpike)

bias = Vector{Param.FloatPrecision}(undef, Param.Ncells)             # total external input to neurons

lastSpike = Array{Float64}(undef, Param.Ncells)  # last time a neuron spiked

plusone = Param.FloatPrecision(1.0)
exactlyzero = Param.FloatPrecision(0.0)
PScale = Param.FloatPrecision(Param.PScale)

vre = Param.FloatPrecision(Param.vre)  # reset voltage

uavg = zeros(Param.FloatPrecision, Param.Ncells)  # average synaptic input
utmp = Matrix{Param.FloatPrecision}(undef, Param.Nsteps - round(Int, 1000/Param.dt), 1000)

PLtot = Param.Lexc + Param.Linh + Param.Lffwd
raug = Matrix{Param.FloatPrecision}(undef, PLtot, Threads.nthreads())
k = Matrix{Param.FloatPrecision}(undef, PLtot, Threads.nthreads())
delta = Matrix{Param.FloatPrecision}(undef, PLtot, Threads.nthreads())
v = Vector{Param.FloatPrecision}(undef, Param.Ncells)       # membrane voltage
noise = Vector{Param.FloatPrecision}(undef, Param.Ncells)   # actual noise added at current time step
sig = fill(Param.FloatPrecision(Param.sig), Param.Ncells)   # std dev of the Gaussian noise

rndFfwd = Vector{Param.FloatPrecision}(undef, Param.Lffwd)  # uniform noise to generate Poisson feed-forward spikes

learn_step = round(Int, Param.learn_every/Param.dt)
