function rls(k, den, e, delta, L, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone, exactlyzero)
    rtrim = r[Px]
    @static if p.PType == Array
        batched_gemv!('N', plusone/PScale, P, rtrim, exactlyzero, k)
    elseif p.PType == Symmetric
        batched_symv!('U', plusone/PScale, P, rtrim, exactlyzero, k)
    elseif p.PType == SymmetricPacked
        batched_spmv!('U', plusone/PScale, P, rtrim, exactlyzero, k)
    end
    batched_dot!(den, rtrim, k)
    den .= plusone ./ (plusone .+ den)
    @static if p.PType == Array
        batched_ger!(-den*PScale, k, k, P)
    elseif p.PType == Symmetric
        batched_syr!('U', -den*PScale, k, P)
    elseif p.PType == SymmetricPacked
        batched_spr!('U', -den*PScale, k, P)
    end
    batched_dot!(e, wpWeightIn, rtrim)
    e .+= synInputBalanced .- xtarg[learn_seq,:]
    delta .= k .* reshape(e, 1, Ncells)
    delta .*= reshape(den, 1, Ncells)
    wpWeightIn .-= delta
    wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    return wpWeightIn, wpWeightOut
end
