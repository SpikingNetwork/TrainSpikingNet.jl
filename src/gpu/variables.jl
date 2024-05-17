TTime = eltype(p.FloatPrecision(p.tau_meme))
TInvTime = eltype(p.FloatPrecision(1/p.tau_meme))
TCurrent = eltype(p.FloatPrecision(p.g))
TCharge = eltype(oneunit(TTime) * oneunit(TCurrent))
TVoltage = eltype(p.FloatPrecision(p.vre))

X_bal = CuArray{TCurrent}(p.X_bal)  # external input

#synaptic time constants
invtau_bale = p.FloatPrecision(1/p.tau_bale)
invtau_bali = p.FloatPrecision(1/p.tau_bali)
if typeof(p.tau_plas)<:Number
    invtau_plas = p.FloatPrecision(1/p.tau_plas)
else
    invtau_plas = CuVector{p.FloatPrecision}(inv.(p.tau_plas))
end

_args = []
for (k,v) in pairs(p.cellModel_args)
    if typeof(v)<:AbstractArray
        push!(_args, k=>CuArray(p.FloatPrecision.(v)))
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
sig = CuArray(sig)
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
learn_nsteps = round(Int, (p.train_time - p.stim_off)/p.learn_every)

_maxTimes = p.maxrate * p.train_time
typeof(p.train_time)<:Real && (_maxTimes /= 1000)
maxTimes = round(Int, _maxTimes)  # maximum number of spikes times to record


bnotrefrac::CuVector{Bool} = Vector(undef, p.Ncells)  # which recurrent neurons are not in the refractory period
bspike::CuVector{Bool} = Vector(undef, p.Ncells)      # which recurrent neurons spiked
bspikeX::CuVector{Bool} = Vector(undef, p.LX)          # which feed-forward neurons spikes


_TTimeInt = Dict(8=>UInt8, 16=>UInt16, 32=>UInt32, 64=>UInt64)[nextpow(2, log2(p.Nsteps))]
_TNoise = p.noise_model==:current ? TCurrent : TVoltage

scratch = Scratch{CuMatrix{_TTimeInt, CUDA.Mem.DeviceBuffer},
                  CuVector{p.IntPrecision, CUDA.Mem.DeviceBuffer},
                  CuVector{TCharge, CUDA.Mem.DeviceBuffer},
                  CuVector{p.FloatPrecision, CUDA.Mem.DeviceBuffer},
                  CuVector{TCurrent, CUDA.Mem.DeviceBuffer},
                  CuMatrix{TCurrent, CUDA.Mem.DeviceBuffer},
                  CuVector{TInvTime, CUDA.Mem.DeviceBuffer},
                  CuMatrix{TInvTime, CUDA.Mem.DeviceBuffer},
                  CuVector{eltype(Float64(p.dt)), CUDA.Mem.DeviceBuffer},
                  CuVector{TVoltage, CUDA.Mem.DeviceBuffer},
                  CuVector{_TNoise, CUDA.Mem.DeviceBuffer}}()

function generate_Pinv!(Pinv, cis, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)
    function kernel(Pinv, cis, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        @inbounds for i=i0:istride:size(Pinv,1)
            for j=j0:jstride:size(Pinv,2)
                for k=1:size(Pinv,3)
                    Pinv[i,j,k] = 0
                    if i==j
                        Pinv[i,j,k] = i<=LX ? penlamFF : penlambda
                    end
                    if i>LX && j>LX
                        Pinv[i,j,k] += penmu * (
                              (wpWeightIn[i,cis[k]] > charge0 && wpWeightIn[j,cis[k]] > charge0) ||
                              (wpWeightIn[i,cis[k]] < charge0 && wpWeightIn[j,cis[k]] < charge0) )
                    end
                end
            end
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(Pinv, cis, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)
    config = launch_configuration(kernel.fun)
    dims = size(Pinv)[1:2]
    xthreads = min(32, dims[1])
    ythreads = min(fld(config.threads, xthreads), cld(prod(dims), xthreads))
    xblocks = min(config.blocks, cld(dims[1], xthreads))
    yblocks = min(cld(config.blocks, xblocks), cld(dims[2], ythreads))
    kernel(Pinv, cis, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda;
           threads=(xthreads,ythreads), blocks=(xblocks<<2,yblocks<<2))
end
