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

const WARP_SIZE = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)
const MAX_GRID_DIM_X = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
const MAX_GRID_DIM_Y = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)

function configurator(kernel, dim1)
    config = launch_configuration(kernel.fun)
    xthreads = min(config.threads, dim1)
    xblocks = min(MAX_GRID_DIM_X, cld(dim1, xthreads))
    return (xthreads,), (xblocks,)
end

function configurator(kernel, dim1, dim2)
    config = launch_configuration(kernel.fun)
    xthreads = min(WARP_SIZE, dim1)
    ythreads = min(fld(config.threads, xthreads), cld(dim1*dim2, xthreads))
    xblocks = min(MAX_GRID_DIM_X, cld(dim1, xthreads))
    yblocks = min(MAX_GRID_DIM_Y, cld(dim2, ythreads))
    return (xthreads, ythreads), (xblocks, yblocks)
end

function generate_Pinv!(Pinv, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)
    function kernel(Pinv, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        i = i0
        @inbounds while i <= size(Pinv,1)
            j = j0
            while j <= size(Pinv,2)
                Pinv[i,j] = 0
                if i==j
                    Pinv[i,j] = i<=LX ? penlamFF : penlambda
                end
                if i>LX && j>LX
                    Pinv[i,j] += penmu * (
                          (wpWeightIn[i,ci] > charge0 && wpWeightIn[j,ci] > charge0) ||
                          (wpWeightIn[i,ci] < charge0 && wpWeightIn[j,ci] < charge0) )
                end
                j += jstride
            end
            i += istride
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(Pinv, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)
    threads, blocks = configurator(kernel, size(Pinv,1), size(Pinv,2))
    kernel(Pinv, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda;
           threads=threads, blocks=blocks)
end
