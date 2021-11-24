mu = Array{p.FloatPrecision}(p.mu)

invtauedecay = p.FloatPrecision(1/p.tauedecay)
invtauidecay = p.FloatPrecision(1/p.tauidecay)
invtaudecay_plastic = Vector{p.FloatPrecision}(inv.(p.taudecay_plastic))

thresh = Vector{p.FloatPrecision}(undef, p.Ncells)
thresh[1:p.Ne] .= p.threshe
thresh[(1+p.Ne):p.Ncells] .= p.threshi

invtau = Vector{p.FloatPrecision}(undef, p.Ncells)
invtau[1:p.Ne] .= 1/p.taue
invtau[(1+p.Ne):p.Ncells] .= 1/p.taui

maxTimes = round(Int,p.maxrate*p.train_time/1000)
times = Array{p.FloatPrecision}(undef, p.Ncells, maxTimes)
ns = Vector{p.IntPrecision}(undef, p.Ncells)

forwardInputsE = Vector{p.FloatPrecision}(undef, p.Ncells)     # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = Vector{p.FloatPrecision}(undef, p.Ncells)     # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = Vector{p.FloatPrecision}(undef, p.Ncells)     # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = Vector{p.FloatPrecision}(undef, p.Ncells) # copy of forwardInputsE from previous time step
forwardInputsIPrev = Vector{p.FloatPrecision}(undef, p.Ncells) # copy of forwardInputsI from previous time step
forwardInputsPPrev = Vector{p.FloatPrecision}(undef, p.Ncells) # copy of forwardInputsP from previous time step
forwardSpike = Vector{p.FloatPrecision}(undef, p.Ncells)       # spikes emitted by each neuron at one time step
forwardSpikePrev = Vector{p.FloatPrecision}(undef, p.Ncells)   # copy of forwardSpike from previous time step

xedecay = Vector{p.FloatPrecision}(undef, p.Ncells)          # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = Vector{p.FloatPrecision}(undef, p.Ncells)          # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = Vector{p.FloatPrecision}(undef, p.Ncells)          # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = Vector{p.FloatPrecision}(undef, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
synInput = Vector{p.FloatPrecision}(undef, p.Ncells)         # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = Vector{p.FloatPrecision}(undef, p.Ncells)                # synapse-filtered spikes (i.e. filtered version of forwardSpike)

bias = Vector{p.FloatPrecision}(undef, p.Ncells)             # total external input to neurons

lastSpike = Array{p.FloatPrecision}(undef, p.Ncells)  # last time a neuron spiked

plusone = p.FloatPrecision(1.0)
exactlyzero = p.FloatPrecision(0.0)

refrac = p.FloatPrecision(p.refrac)
vre = p.FloatPrecision(p.vre)

uavg = zeros(p.FloatPrecision, p.Ncells)
utmp = Matrix{p.FloatPrecision}(undef, p.Nsteps - Int(1000/p.dt), 1000)

k = Matrix{p.FloatPrecision}(undef, 2*p.L, Threads.nthreads())
v = Vector{p.FloatPrecision}(undef, p.Ncells)
noise = Vector{p.FloatPrecision}(undef, p.Ncells)
sig = fill(p.FloatPrecision(p.sig0), p.Ncells)

dt = p.FloatPrecision(p.dt)
learn_step = round(Int, p.learn_every/dt)
