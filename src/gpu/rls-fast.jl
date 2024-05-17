function rls(itask,
             raug::CuMatrix{T}, k, vPv, den, e, delta, r, rX, P, u_bal,
             utarg, LX, learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero, PScale) where T

    @static p.LX>0 && (raug[1:LX,:] .= rX)
    raug[LX+1:end,:] = @view r[0x1 .+ wpIndexIn]
    _raug = T<:Real ? raug : ustrip(raug)

    @static if p.PType == Array
        batched_gemv!('N', plusone/PScale, P, _raug, exactlyzero, k)
    elseif p.PType == Symmetric
        batched_symv!('U', plusone/PScale, P, _raug, exactlyzero, k)
    elseif p.PType == SymmetricPacked
        batched_spmv!('U', plusone/PScale, P, _raug, exactlyzero, k)
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

    @static if p.PType == Array
        batched_ger!(-_den*PScale, k, k, P)
    elseif p.PType == Symmetric
        batched_syr!('U', -_den*PScale, k, P)
    elseif p.PType == SymmetricPacked
        batched_spr!('U', -_den*PScale, k, P)
    end

    batched_dot!(_e, _wpWeightIn, _raug[LX+1:end,:])
    e .+= u_bal .- @view utarg[learn_seq,:,itask]
    @static p.LX>0 && (e .+= wpWeightX * rX)
    delta .= e' .* k
    @static if !p.benchmark
        wpWeightIn .-= @view delta[LX+1:end,:]
        @static p.LX>0 && (wpWeightX .-= (@view delta[1:LX,:])')
    end
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    return wpWeightIn, wpWeightOut
end
