mu = CuVector{p.FloatPrecision}(undef, p.Ncells)
mu[1:p.Ne] = (p.muemax-p.muemin)*rand(p.Ne) .+ p.muemin
mu[(p.Ne+1):p.Ncells] = (p.muimax-p.muimin)*rand(p.Ni) .+ p.muimin

thresh = CuVector{p.FloatPrecision}(undef, p.Ncells)
thresh[1:p.Ne] .= p.threshe
thresh[(1+p.Ne):p.Ncells] .= p.threshi

invtau = CuVector{p.FloatPrecision}(undef, p.Ncells)
invtau[1:p.Ne] .= 1/p.taue
invtau[(1+p.Ne):p.Ncells] .= 1/p.taui

ns = CuVector{p.IntPrecision}(undef, p.Ncells)

forwardInputsE = CuVector{p.FloatPrecision}(undef, p.Ncells+1)  # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = CuVector{p.FloatPrecision}(undef, p.Ncells+1)  # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = CuVector{p.FloatPrecision}(undef, p.Ncells+1)  # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = CuVector{p.FloatPrecision}(undef, p.Ncells+1) # copy of forwardInputsE from previous time step
forwardInputsIPrev = CuVector{p.FloatPrecision}(undef, p.Ncells+1) # copy of forwardInputsI from previous time step
forwardInputsPPrev = CuVector{p.FloatPrecision}(undef, p.Ncells+1) # copy of forwardInputsP from previous time step
forwardSpike = CuVector{p.FloatPrecision}(undef, p.Ncells) # spikes emitted by each neuron at one time step
forwardSpikePrev = CuVector{p.FloatPrecision}(undef, p.Ncells) # copy of forwardSpike from previous time step

xedecay = CuVector{p.FloatPrecision}(undef, p.Ncells)  # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = CuVector{p.FloatPrecision}(undef, p.Ncells)  # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = CuVector{p.FloatPrecision}(undef, p.Ncells)  # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = CuVector{p.FloatPrecision}(undef, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
synInput = CuVector{p.FloatPrecision}(undef, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = CuVector{p.FloatPrecision}(undef, p.Ncells)      # synapse-filtered spikes (i.e. filtered version of forwardSpike)

bias = CuVector{p.FloatPrecision}(undef, p.Ncells)   # total external input to neurons
lastSpike = CuArray{p.FloatPrecision}(undef, p.Ncells)  # last time a neuron spiked

bnotrefrac = CuVector{Bool}(undef, p.Ncells)
bspike = CuVector{Bool}(undef, p.Ncells)
plusone = p.FloatPrecision(1.0)
minusone = p.FloatPrecision(-1.0)

refrac = p.FloatPrecision(p.refrac)
vre = p.FloatPrecision(p.vre)

k = CuArray{p.FloatPrecision}(undef, 2*p.L, 1, p.Ncells)
den = CuArray{p.FloatPrecision}(undef, 1, 1, p.Ncells)
e = CuArray{p.FloatPrecision}(undef, 1, 1, p.Ncells)
delta = CuArray{p.FloatPrecision}(undef, 2*p.L, 1, p.Ncells)
v = CuVector{p.FloatPrecision}(undef, p.Ncells)
