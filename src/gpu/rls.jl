function rls(itask, raug, k, den, e, delta, L, Ncells, Lei, r, s, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightFfwd, wpWeightIn, wpWeightOut, plusone, minusone, exactlyzero)
    raug[1:Lei,:] = @view r[Px]
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
    e .+= synInputBalanced .- @view xtarg[learn_seq,:,itask]
    @static p.Lffwd>0 && (e .+= wpWeightFfwd*s)
    delta .= e' .* k .* den'
    wpWeightIn .-= @view delta[1:Lei,:]
    @static p.Lffwd>0 && (wpWeightFfwd .-= (@view delta[Lei+1:end,:])')
    wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut
end
