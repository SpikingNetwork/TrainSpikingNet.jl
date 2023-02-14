function rls(itask,
             raug, k, den, e, delta, r, rX, P, u_bal,
             utarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, minusone, exactlyzero, PScale)

    lenrX = length(rX)
    @static p.LX>0 && (raug[1:lenrX,:] .= rX)
    raug[lenrX+1:end,:] = @view r[0x1 .+ wpIndexIn]

    @static if p.PType == Array
        batched_gemv!('N', plusone/PScale, P, raug, exactlyzero, k)
    elseif p.PType == Symmetric
        batched_symv!('U', plusone/PScale, P, raug, exactlyzero, k)
    elseif p.PType == SymmetricPacked
        batched_spmv!('U', plusone/PScale, P, raug, exactlyzero, k)
    end

    batched_dot!(den, raug, k)
    den .= plusone ./ (plusone .+ den)

    @static if p.PType == Array
        batched_ger!(-den*PScale, k, k, P)
    elseif p.PType == Symmetric
        batched_syr!('U', -den*PScale, k, P)
    elseif p.PType == SymmetricPacked
        batched_spr!('U', -den*PScale, k, P)
    end

    batched_dot!(e, wpWeightIn, raug[lenrX+1:end,:])
    e .+= u_bal .- @view utarg[learn_seq,:,itask]
    @static p.LX>0 && (e .+= wpWeightX*rX)
    delta .= e' .* k .* den'
    wpWeightIn .-= @view delta[lenrX+1:end,:]
    @static p.LX>0 && (wpWeightX .-= (@view delta[1:lenrX,:])')
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    return wpWeightIn, wpWeightOut
end
