function rls(itask,
             raug::CuMatrix{T}, k, k2, rrXg, vPv, den, e, delta, Ncells, r, rX,
             Pinv, pivot, pivot64, workspace, u_bal, utarg,
             rrXhistory, charge0, LX, penmu, penlamFF, penlambda,
             learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero, PScale) where T

    copyto_raug(raug, LX, rX, r, wpIndexIn)
    _raug = T<:Real ? raug : ustrip(raug)

    for ci = 1:Ncells
        generate_Pinv!(Pinv, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)

        # Pinv += rrXhistory' * rrXhistory
        for rrXi in eachcol(rrXhistory)
            copyto_rrXg(rrXg, rrXi, LX, wpIndexIn, ci)
            @static if p.PType == Array
                CUBLAS.ger!(plusone, rrXg, rrXg, Pinv)
            elseif p.PType == Symmetric
                CUBLAS.syr!('U', plusone, rrXg, Pinv)
            end
        end

        # k = Pinv \ (_raug / PScale)
        k .= _raug ./ PScale
    	@static if p.PType == Array
            #ldiv!(view(k,:,ci), lu!(Pinv), view(_raug,:,ci) ./ PScale)
            CUSOLVER.getrf!(Pinv, pivot, workspace)
            CUSOLVER.getrs!('N', Pinv, pivot, view(k,:,ci))
    	elseif p.PType == Symmetric
            #ldiv!(view(k,:,ci), bunchkaufman!(Symmetric(Pinv)), view(_raug,:,ci) / PScale)
            CUSOLVER.sytrf!('U', Pinv, pivot, workspace)
            pivot64 .= pivot
            CUSOLVER.sytrs!('U', Pinv, pivot64, view(k2,:,:,ci))
    	end
    end

    if T<:Real
        batched_dot!(den, _raug, k)
        den .= plusone ./ (plusone .+ den)
        _den = den
        _e = e
        _wpWeightIn = wpWeightIn
    else
        _vPv = ustrip(vPv)
        batched_dot!(_vPv, _raug, k)
        den .= plusone ./ (oneunit.(vPv) .+ vPv)
        _den = ustrip(den)
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

    push!(rrXhistory, (@view rrXhistory[:,end]))
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

        @inbounds for i=i0:istride:size(raug,1), j=j0:jstride:size(raug,2)
            raug[i,j] = i<=LX ? rX[i] : r[wpIndexIn[i,j]]
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(raug, LX, rX, r, wpIndexIn)
    threads, blocks = configurator(kernel, size(raug,1), size(raug,2))
    kernel(raug, LX, rX, r, wpIndexIn; threads=threads, blocks=blocks)
end

function copyto_rrXg(rrXg, rrXi, LX, wpIndexIn, ci)

    function kernel(rrXg, rrXi, LX, wpIndexIn, ci)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        istride = blockDim().x * gridDim().x

        @inbounds for i=i0:istride:size(rrXg,1)
            rrXg[i] = i<=LX ? rrXi[i] : rrXi[wpIndexIn[i,ci]]
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(rrXg, rrXi, LX, wpIndexIn, ci)
    threads, blocks = configurator(kernel, size(rrXg,1))
    kernel(rrXg, rrXi, LX, wpIndexIn, ci; threads=threads, blocks=blocks)
end
