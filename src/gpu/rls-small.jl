function rls(itask,
             raug::CuMatrix{T}, k, k2, rrXg, e, delta, Ncells, PComputeN, r, rX,
             Pinv, u_bal, utarg,
             rrXhistory, charge0, LX, penmu, penlamFF, penlambda,
             learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero, PScale) where T

    @static p.LX>0 && (raug[1:LX,:] .= rX)
    raug[LX+1:end,:] = @view r[0x1 .+ wpIndexIn]
    _raug = T<:Real ? raug : ustrip(raug)

    for ci = 1:PComputeN:Ncells
        generate_Pinv!(Pinv, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)

        # Pinv += rrXhistory' * rrXhistory
        for rrXi in eachcol(rrXhistory)
            copyto!(rrXg, 1, rrXi, 1, LX)
            rrXg[LX+1:end,:] .= view(rrXi, 0x1 .+ view(wpIndexIn,:, ci:ci+PComputeN-1))
            @static if p.PType == Array
                batched_ger!(plusone, rrXg, rrXg, Pinv)
            end
        end

        # k = Pinv \ (_raug / PScale)
        k .= _raug / PScale
        @static if p.PType == Array
            ipiv,_ = CUDA.CUBLAS.getrf_strided_batched!(Pinv, true)
            CUDA.CUBLAS.getrs_strided_batched!('N', Pinv, ipiv, view(k2,:,:,ci:ci+PComputeN-1))
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
