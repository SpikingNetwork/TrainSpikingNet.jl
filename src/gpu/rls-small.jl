function rls(itask,
             raug::CuMatrix{T}, k, k2, rrXg, e, delta, Ncells, PComputeN, r, rX,
             Pinv, pivot, info, u_bal, utarg,
             rrXhistory, charge0, LX, penmu, penlamFF, penlambda,
             learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero, PScale) where T

    copyto_raug(raug, LX, rX, r, wpIndexIn)
    _raug = T<:Real ? raug : ustrip(raug)

    for ci = 1:PComputeN:Ncells
        generate_Pinv!(Pinv, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)

        # Pinv += rrXhistory' * rrXhistory
        for rrXi in eachcol(rrXhistory)
            copyto_rrXg(rrXg, rrXi, LX, wpIndexIn, ci)
            @static if p.PType == Array
                batched_ger!(plusone, rrXg, rrXg, Pinv)
            end
        end

        # k = Pinv \ (_raug / PScale)
        k .= _raug ./ PScale
        @static if p.PType == Array
            CUDA.CUBLAS.getrf_strided_batched!(Pinv, pivot, info)
            CUDA.CUBLAS.getrs_strided_batched!('N', Pinv, pivot, view(k2,:,:,ci:ci+PComputeN-1))
        end
    end

    if T<:Real
        _e = e
        _wpWeightIn = wpWeightIn
    else
        _e = ustrip(e)
        _wpWeightIn = ustrip(wpWeightIn)
    end

    batched_dot!(_e, _wpWeightIn, (@view _raug[LX+1:end,:]))
    e .+= u_bal .- @view utarg[learn_seq,:,itask]
    @static p.LX>0 && (e .+= wpWeightX * rX)
    delta .= e' .* k
    @static if !p.benchmark
        wpWeightIn .-= @view delta[LX+1:end,:]
        @static p.LX>0 && (wpWeightX .-= (@view delta[1:LX,:])')
    end
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    push!(rrXhistory, 0)
    @static p.LX>0 && (rrXhistory[1:LX, end] = rX)
    rrXhistory[LX+1:end, end] = r

    return wpWeightIn, wpWeightOut
end

function copyto_raug(raug, LX, rX, r, wpIndexIn)

    function kernel(raug, LX, rX, r, wpIndexIn)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        @inbounds for i=i0:istride:size(raug,1)
            for j=j0:jstride:size(raug,2)
                raug[i,j] = i<=LX ? rX[i] : r[wpIndexIn[i,j]]
            end
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(raug, LX, rX, r, wpIndexIn)
    config = launch_configuration(kernel.fun)
    dims = size(raug)
    xthreads = min(32, dims[1])
    ythreads = min(fld(config.threads, xthreads), cld(prod(dims), xthreads))
    xblocks = min(config.blocks, cld(dims[1], xthreads))
    yblocks = min(cld(config.blocks, xblocks), cld(dims[2], ythreads))
    kernel(raug, LX, rX, r, wpIndexIn;
           threads=(xthreads,ythreads), blocks=(xblocks<<2,yblocks<<2))
end

function copyto_rrXg(rrXg, rrXi, LX, wpIndexIn, ci)

    function kernel(rrXg, rrXi, LX, wpIndexIn, ci)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        @inbounds for i=i0:istride:size(rrXg,1)
            for j=j0:jstride:size(rrXg,2)
                rrXg[i,j] = i<=LX ? rrXi[i] : rrXi[wpIndexIn[i,ci+j-1]]
            end
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(rrXg, rrXi, LX, wpIndexIn, ci)
    config = launch_configuration(kernel.fun)
    dims = size(rrXg)
    xthreads = min(32, dims[1])
    ythreads = min(fld(config.threads, xthreads), cld(prod(dims), xthreads))
    xblocks = min(config.blocks, cld(dims[1], xthreads))
    yblocks = min(cld(config.blocks, xblocks), cld(dims[2], ythreads))
    kernel(rrXg, rrXi, LX, wpIndexIn, ci;
           threads=(xthreads,ythreads), blocks=(xblocks<<2,yblocks<<2))
end
