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

plusone = p.FloatPrecision(1.0)
exactlyzero = p.FloatPrecision(0.0)

vre = p.FloatPrecision(p.vre)  # reset voltage

learn_step = round(Int, p.learn_every/p.dt)

u0_skip_steps = round(Int, p.u0_skip_time/p.dt)

_maxTimes = p.maxrate * p.train_time
typeof(p.train_time)<:Real && (_maxTimes /= 1000)
maxTimes = round(Int, _maxTimes)  # maximum number of spikes times to record


uavg::Vector{TCurrent} = zeros(TCurrent, p.Ncells)  # average synaptic input
ustd::Matrix{TCurrent} = Matrix(undef, p.Nsteps - u0_skip_steps, p.u0_ncells)


_TTimeInt = Dict(8=>UInt8, 16=>UInt16, 32=>UInt32, 64=>UInt64)[nextpow(2, log2(p.Nsteps))]
_TNoise = p.noise_model==:current ? TCurrent : TVoltage

scratch = Scratch{Matrix{_TTimeInt},
                  Vector{p.IntPrecision},
                  Vector{TCharge},
                  Vector{p.FloatPrecision},
                  Vector{TCurrent},
                  Vector{TInvTime},
                  Vector{eltype(Float64(p.dt))},
                  Vector{TVoltage},
                  Vector{_TNoise}}()
