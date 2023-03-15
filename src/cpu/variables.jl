TTime = eltype(p.FloatPrecision(p.tau_meme))
TInvTime = eltype(p.FloatPrecision(1/p.tau_meme))
TCurrent = eltype(p.FloatPrecision(p.g))
TCharge = eltype(oneunit(TTime) * oneunit(TCurrent))
TTime <: Real || (HzUnit = unit(p.maxrate))
TVoltage = eltype(p.FloatPrecision(p.vre))

X_bal = Array{TCurrent}(p.X_bal)  # external input

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
        push!(_args, k=>p.FloatPrecision.(v))
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

_maxTimes = p.maxrate * p.train_time
typeof(p.train_time)<:Real && (_maxTimes /= 1000)
maxTimes = round(Int, _maxTimes)  # maximum number of spikes times to record
_times_precision = Dict(8=>UInt8, 16=>UInt16, 32=>UInt32, 64=>UInt64)[nextpow(2, log2(p.Nsteps))]
times = Array{_times_precision}(undef, p.Ncells, maxTimes)  # times of recurrent spikes throughout trial
ns = Vector{p.IntPrecision}(undef, p.Ncells)            # number of recurrent spikes in trial
timesX = Array{_times_precision}(undef, p.LX, maxTimes)     # times of feed-forward spikes throughout trial
nsX = Vector{p.IntPrecision}(undef, p.LX)               # number of feed-forward spikes in trial

inputsE = Vector{TCharge}(undef, p.Ncells)      # excitatory synaptic currents to neurons via balanced connections at one time step
inputsI = Vector{TCharge}(undef, p.Ncells)      # inhibitory synaptic currents to neurons via balanced connections at one time step
inputsP = Vector{TCharge}(undef, p.Ncells)      # synaptic currents to neurons via plastic connections at one time step
inputsEPrev = Vector{TCharge}(undef, p.Ncells)  # copy of inputsE from previous time step
inputsIPrev = Vector{TCharge}(undef, p.Ncells)  # copy of inputsI from previous time step
inputsPPrev = Vector{TCharge}(undef, p.Ncells)  # copy of inputsP from previous time step
spikes = Vector{p.FloatPrecision}(undef, p.Ncells)       # spikes emitted by each recurrent neuron at one time step
spikesPrev = Vector{p.FloatPrecision}(undef, p.Ncells)   # copy of spike from previous time step
spikesX = Vector{p.FloatPrecision}(undef, p.LX)          # spikes emitted by each feed-forward neuron at one time step
spikesXPrev = Vector{p.FloatPrecision}(undef, p.LX)      # copy of spikesX from previous time step

u_bale = Vector{TCurrent}(undef, p.Ncells)   # synapse-filtered excitatory current (i.e. filtered version of inputsE)
u_bali = Vector{TCurrent}(undef, p.Ncells)   # synapse-filtered inhibitory current (i.e. filtered version of inputsI)
uX_plas = Vector{TCurrent}(undef, p.Ncells)  # synapse-filtered plastic current (i.e. filtered version of inputsP)
u_bal = Vector{TCurrent}(undef, p.Ncells)    # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
u = Vector{TCurrent}(undef, p.Ncells)        # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
r = Vector{TInvTime}(undef, p.Ncells)        # synapse-filtered recurrent spikes (i.e. filtered version of spike)
rX = Vector{TInvTime}(undef, p.LX)           # synapse-filtered feed-forward spikes (i.e. filtered version of spikesX)

X = Vector{TCurrent}(undef, p.Ncells)        # total external input to neurons
lastSpike = Array{eltype(Float64(p.dt))}(undef, p.Ncells)    # last time a neuron spiked

plusone = p.FloatPrecision(1.0)
exactlyzero = p.FloatPrecision(0.0)

vre = p.FloatPrecision(p.vre)  # reset voltage

u0_skip_steps = round(Int, p.u0_skip_time/p.dt)
uavg = zeros(TCurrent, p.Ncells)  # average synaptic input
ustd = Matrix{TCurrent}(undef, p.Nsteps - u0_skip_steps, p.u0_ncells)

v = Vector{TVoltage}(undef, p.Ncells)      # membrane voltage
noise = Vector{p.noise_model==:current ? TCurrent : TVoltage}(undef, p.Ncells)  # actual noise added at each time step

rndX = Vector{p.FloatPrecision}(undef, p.LX)    # uniform noise to generate Poisson feed-forward spikes

learn_step = round(Int, p.learn_every/p.dt)
