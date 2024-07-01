function rls(itask,
             raug::CuVector{T}, k, k2, rrXg, delta, Ncells, r, rX,
             Pinv, pivot, info, u_bal, utarg,
             rrXhistory, charge0, LX, penmu, penlamFF, penlambda,
             learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero, PScale) where T

    for ci = 1:Ncells
        copyto_raug(raug, LX, rX, r, wpIndexIn, ci)

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

        # k = Pinv \ (raug / PScale)
        k .= raug ./ PScale
    	@static if p.PType == Array
    		CUSOLVER.getrf!(Pinv, pivot)
    		CUSOLVER.getrs!('N', Pinv, pivot, k)
    	elseif p.PType == Symmetric
    		CUSOLVER.sytrf!('U', Pinv)
    		CUSOLVER.sytrs!('U', Pinv, k2)
    	end

        CUDA.@allowscalar e = view(wpWeightIn,:,ci)' * (@view raug[LX+1:end]) + u_bal[ci] - utarg[learn_seq,ci,itask]
        @static p.LX>0 && (e += view(wpWeightX,ci,:)' * rX)
        delta .= e .* k
        @static if !p.benchmark
            view(wpWeightIn,:,ci) .-= @view delta[LX+1:end]
            @static p.LX>0 && (wpWeightX[ci,:] .-= (@view delta[1:LX])')
        end
    end

    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    push!(rrXhistory, 0)
    @static p.LX>0 && (rrXhistory[1:LX, end] = rX)
    rrXhistory[LX+1:end, end] = r

    return wpWeightIn, wpWeightOut
end

function copyto_raug(raug, LX, rX, r, wpIndexIn, ci)

    function kernel(raug, LX, rX, r, wpIndexIn, ci)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        istride = blockDim().x * gridDim().x

        @inbounds for i=i0:istride:size(raug,1)
            raug[i] = i<=LX ? rX[i] : r[wpIndexIn[i,ci]]
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(raug, LX, rX, r, wpIndexIn, ci)
    threads, blocks = configurator(kernel, size(raug,1))
    kernel(raug, LX, rX, r, wpIndexIn, ci; threads=threads, blocks=blocks)
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
