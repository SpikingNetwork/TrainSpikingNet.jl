function rls(itask,
             raug, k, den, e, delta, Ncells, Lei, r, s, P, u_bal,
             utarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, minusone, exactlyzero, PScale)

    raug[1:Lei,:] = @view r[wpIndexIn]
    raug[Lei+1:end,:] .= s

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

    batched_dot!(e, wpWeightIn, raug[1:Lei,:])
    e .+= u_bal .- @view utarg[learn_seq,:,itask]
    @static p.LX>0 && (e .+= wpWeightX*s)
    delta .= e' .* k .* den'
    wpWeightIn .-= @view delta[1:Lei,:]
    @static p.LX>0 && (wpWeightX .-= (@view delta[Lei+1:end,:])')
    wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut
end
