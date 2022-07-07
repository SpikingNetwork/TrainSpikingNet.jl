mu = Array{p.FloatPrecision}(p.mu)

invtauedecay = p.FloatPrecision(1/p.tauedecay)
invtauidecay = p.FloatPrecision(1/p.tauidecay)
if typeof(p.taudecay_plastic)<:Number
    invtaudecay_plastic = p.FloatPrecision(1/p.taudecay_plastic)
else
    invtaudecay_plastic = Vector{p.FloatPrecision}(inv.(p.taudecay_plastic))
end

thresh = Vector{p.FloatPrecision}(undef, p.Ncells)
thresh[1:p.Ne] .= p.threshe
thresh[(1+p.Ne):p.Ncells] .= p.threshi

tau = Vector{p.FloatPrecision}(undef, p.Ncells)
tau[1:p.Ne] .= p.taue
tau[(1+p.Ne):p.Ncells] .= p.taui

maxTimes = round(Int,p.maxrate*p.train_time/1000)
times = Array{Float64}(undef, p.Ncells, maxTimes)
ns = Vector{p.IntPrecision}(undef, p.Ncells)
times_ffwd = Array{Float64}(undef, p.Lffwd, maxTimes)
ns_ffwd = Vector{p.IntPrecision}(undef, p.Lffwd)

forwardInputsE = Vector{p.FloatPrecision}(undef, p.Ncells)     # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = Vector{p.FloatPrecision}(undef, p.Ncells)     # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = Vector{p.FloatPrecision}(undef, p.Ncells)     # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = Vector{p.FloatPrecision}(undef, p.Ncells) # copy of forwardInputsE from previous time step
forwardInputsIPrev = Vector{p.FloatPrecision}(undef, p.Ncells) # copy of forwardInputsI from previous time step
forwardInputsPPrev = Vector{p.FloatPrecision}(undef, p.Ncells) # copy of forwardInputsP from previous time step
forwardSpike = Vector{p.FloatPrecision}(undef, p.Ncells)       # spikes emitted by each neuron at one time step
forwardSpikePrev = Vector{p.FloatPrecision}(undef, p.Ncells)   # copy of forwardSpike from previous time step
ffwdSpike = Vector{p.FloatPrecision}(undef, p.Lffwd)
ffwdSpikePrev = Vector{p.FloatPrecision}(undef, p.Lffwd)

xedecay = Vector{p.FloatPrecision}(undef, p.Ncells)          # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = Vector{p.FloatPrecision}(undef, p.Ncells)          # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = Vector{p.FloatPrecision}(undef, p.Ncells)          # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = Vector{p.FloatPrecision}(undef, p.Ncells) # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
synInput = Vector{p.FloatPrecision}(undef, p.Ncells)         # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = Vector{p.FloatPrecision}(undef, p.Ncells)                # synapse-filtered spikes (i.e. filtered version of forwardSpike)
s = Vector{p.FloatPrecision}(undef, p.Lffwd)

bias = Vector{p.FloatPrecision}(undef, p.Ncells)             # total external input to neurons

lastSpike = Array{Float64}(undef, p.Ncells)  # last time a neuron spiked

plusone = p.FloatPrecision(1.0)
exactlyzero = p.FloatPrecision(0.0)
PScale = p.FloatPrecision(p.PScale)

vre = p.FloatPrecision(p.vre)

uavg = zeros(p.FloatPrecision, p.Ncells)
utmp = Matrix{p.FloatPrecision}(undef, p.Nsteps - round(Int, 1000/p.dt), 1000)

raug = Matrix{p.FloatPrecision}(undef, p.Lexc+p.Linh+p.Lffwd, Threads.nthreads())
k = Matrix{p.FloatPrecision}(undef, p.Lexc+p.Linh+p.Lffwd, Threads.nthreads())
delta = Matrix{p.FloatPrecision}(undef, p.Lexc+p.Linh+p.Lffwd, Threads.nthreads())
v = Vector{p.FloatPrecision}(undef, p.Ncells)
noise = Vector{p.FloatPrecision}(undef, p.Ncells)
sig = fill(p.FloatPrecision(p.sig0), p.Ncells)

rndFfwd = Vector{p.FloatPrecision}(undef, p.Lffwd)

learn_step = round(Int, p.learn_every/p.dt)
