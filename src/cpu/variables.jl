invtauedecay = p.FloatPrecision(1/p.tauedecay)
invtauidecay = p.FloatPrecision(1/p.tauidecay)
invtaudecay_plastic = p.FloatPrecision(1/p.taudecay_plastic)

mu = zeros(p.FloatPrecision, p.Ncells)
mu[1:p.Ne] = (p.muemax-p.muemin)*rand(p.Ne) .+ p.muemin
mu[(p.Ne+1):p.Ncells] = (p.muimax-p.muimin)*rand(p.Ni) .+ p.muimin

thresh = zeros(p.FloatPrecision, p.Ncells)
thresh[1:p.Ne] .= p.threshe
thresh[(1+p.Ne):p.Ncells] .= p.threshi

invtau = zeros(p.FloatPrecision, p.Ncells)
invtau[1:p.Ne] .= 1/p.taue
invtau[(1+p.Ne):p.Ncells] .= 1/p.taui

ns = zeros(p.IntPrecision, p.Ncells)

forwardInputsE = zeros(p.FloatPrecision, p.Ncells)     # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = zeros(p.FloatPrecision, p.Ncells)     # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = zeros(p.FloatPrecision, p.Ncells)     # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = zeros(p.FloatPrecision, p.Ncells) # copy of forwardInputsE from previous time step
forwardInputsIPrev = zeros(p.FloatPrecision, p.Ncells) # copy of forwardInputsI from previous time step
forwardInputsPPrev = zeros(p.FloatPrecision, p.Ncells) # copy of forwardInputsP from previous time step
forwardSpike = zeros(p.FloatPrecision, p.Ncells)       # spikes emitted by each neuron at one time step
forwardSpikePrev = zeros(p.FloatPrecision, p.Ncells)   # copy of forwardSpike from previous time step

xedecay = zeros(p.FloatPrecision, p.Ncells)          # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = zeros(p.FloatPrecision, p.Ncells)          # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = zeros(p.FloatPrecision, p.Ncells)          # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = zeros(p.FloatPrecision, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
synInput = zeros(p.FloatPrecision, p.Ncells)         # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = zeros(p.FloatPrecision, p.Ncells)                # synapse-filtered spikes (i.e. filtered version of forwardSpike)

bias = zeros(p.FloatPrecision, p.Ncells)             # total external input to neurons

lastSpike = Array{p.FloatPrecision}(undef, p.Ncells)  # last time a neuron spiked

plusone = p.FloatPrecision(1.0)

refrac = p.FloatPrecision(p.refrac)
vre = p.FloatPrecision(p.vre)

uavg = zeros(p.Ncells)
utmp = zeros(p.Nsteps - Int(1000/p.dt),1000)

k = Matrix{p.FloatPrecision}(undef, 2*p.L, p.Ncells)
v = Vector{p.FloatPrecision}(undef, p.Ncells)

dt = p.FloatPrecision(p.dt)
