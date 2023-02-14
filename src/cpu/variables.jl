X_bal = Array{p.FloatPrecision}(p.X_bal)  # external input

#synaptic time constants
invtau_bale = p.FloatPrecision(1/p.tau_bale)
invtau_bali = p.FloatPrecision(1/p.tau_bali)
if typeof(p.tau_plas)<:Number
    invtau_plas = p.FloatPrecision(1/p.tau_plas)
else
    invtau_plas = Vector{p.FloatPrecision}(inv.(p.tau_plas))
end

_args = []
for (k,v) in pairs(p.cellModel_args)
    if typeof(v)<:AbstractArray
        push!(_args, k=>Array{p.FloatPrecision}(v))
    else
        push!(_args, k=>v)
    end
end
cellModel_args = (; _args...)
_args = nothing

if typeof(p.sig)<:Number
    sig = fill(p.FloatPrecision(p.sig), p.Ncells)
else
    sig = p.FloatPrecision.(p.sig)
end
if any(sig .> 0)
    if p.noise_model==:voltage
        sig .*= sqrt(p.dt) / sqrt.(p.tau_meme)
    elseif p.noise_model==:current
        sig .*= sqrt.(p.tau_meme) / sqrt(p.dt)
    else
        error("noise_model must :voltage or :current")
    end
end

maxTimes = round(Int, p.maxrate * p.train_time / 1000)  # maximum number of spikes times to record
_times_precision = Dict(8=>UInt8, 16=>UInt16, 32=>UInt32, 64=>UInt64)[nextpow(2, log2(p.Nsteps))]
times = Array{_times_precision}(undef, p.Ncells, maxTimes)  # times of recurrent spikes throughout trial
ns = Vector{p.IntPrecision}(undef, p.Ncells)            # number of recurrent spikes in trial
timesX = Array{_times_precision}(undef, p.LX, maxTimes)     # times of feed-forward spikes throughout trial
nsX = Vector{p.IntPrecision}(undef, p.LX)               # number of feed-forward spikes in trial

inputsE = Vector{p.FloatPrecision}(undef, p.Ncells)      # excitatory synaptic currents to neurons via balanced connections at one time step
inputsI = Vector{p.FloatPrecision}(undef, p.Ncells)      # inhibitory synaptic currents to neurons via balanced connections at one time step
inputsP = Vector{p.FloatPrecision}(undef, p.Ncells)      # synaptic currents to neurons via plastic connections at one time step
inputsEPrev = Vector{p.FloatPrecision}(undef, p.Ncells)  # copy of inputsE from previous time step
inputsIPrev = Vector{p.FloatPrecision}(undef, p.Ncells)  # copy of inputsI from previous time step
inputsPPrev = Vector{p.FloatPrecision}(undef, p.Ncells)  # copy of inputsP from previous time step
spikes = Vector{p.FloatPrecision}(undef, p.Ncells)       # spikes emitted by each recurrent neuron at one time step
spikesPrev = Vector{p.FloatPrecision}(undef, p.Ncells)   # copy of spike from previous time step
spikesX = Vector{p.FloatPrecision}(undef, p.LX)          # spikes emitted by each feed-forward neuron at one time step
spikesXPrev = Vector{p.FloatPrecision}(undef, p.LX)      # copy of spikesX from previous time step

u_bale = Vector{p.FloatPrecision}(undef, p.Ncells)   # synapse-filtered excitatory current (i.e. filtered version of inputsE)
u_bali = Vector{p.FloatPrecision}(undef, p.Ncells)   # synapse-filtered inhibitory current (i.e. filtered version of inputsI)
uX_plas = Vector{p.FloatPrecision}(undef, p.Ncells)  # synapse-filtered plastic current (i.e. filtered version of inputsP)
u_bal = Vector{p.FloatPrecision}(undef, p.Ncells)    # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
u = Vector{p.FloatPrecision}(undef, p.Ncells)        # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
r = Vector{p.FloatPrecision}(undef, p.Ncells)        # synapse-filtered recurrent spikes (i.e. filtered version of spike)
rX = Vector{p.FloatPrecision}(undef, p.LX)           # synapse-filtered feed-forward spikes (i.e. filtered version of spikesX)

X = Vector{p.FloatPrecision}(undef, p.Ncells)  # total external input to neurons
lastSpike = Array{Float64}(undef, p.Ncells)           # last time a neuron spiked

plusone = p.FloatPrecision(1.0)
exactlyzero = p.FloatPrecision(0.0)

vre = p.FloatPrecision(p.vre)  # reset voltage

uavg = zeros(Float64, p.Ncells)  # average synaptic input
ustd = Matrix{Float64}(undef, p.Nsteps - round(Int, 1000/p.dt), 1000)

v = Vector{p.FloatPrecision}(undef, p.Ncells)      # membrane voltage
noise = Vector{p.FloatPrecision}(undef, p.Ncells)  # actual noise added at current time step

rndX = Vector{p.FloatPrecision}(undef, p.LX)    # uniform noise to generate Poisson feed-forward spikes

learn_step = round(Int, p.learn_every/p.dt)
