function rls(k, den, e, delta, L, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone, exactlyzero)
    @static if p.PType == Array
        rtrim = reshape(r[Px], 2*L, 1, Ncells)
    else
        rtrim = r[Px]
    end
    @static if p.PType == Array
        batched_mul!(k, P, rtrim)
    elseif p.PType == Symmetric
        batched_symv!('U', plusone, P, rtrim, exactlyzero, k)
    elseif p.PType == SymmetricPacked
        batched_spmv!('U', plusone, P, rtrim, exactlyzero, k)
    end
    @static if p.PType == Array
        batched_mul!(den, batched_transpose(rtrim), k)
    else
        batched_dot!(den, rtrim, k)
    end
    den .= plusone ./ (plusone .+ den)
    @static if p.PType == Array
        delta .= k .* den
        batched_mul!(P, k, batched_transpose(delta), minusone, plusone)
    elseif p.PType == Symmetric
        batched_syr!('U', -den, k, P)
    elseif p.PType == SymmetricPacked
        batched_spr!('U', -den, k, P)
    end
    @static if p.PType == Array
        batched_mul!(e, batched_transpose(wpWeightIn), rtrim)
        e .+= reshape(synInputBalanced .- xtarg[learn_seq,:], 1,1,Ncells)
        delta .= k .* e
        delta .*= den
    else
        batched_dot!(e, wpWeightIn, rtrim)
        e .+= synInputBalanced .- xtarg[learn_seq,:]
        delta .= k .* reshape(e, 1, Ncells)
        delta .*= reshape(den, 1, Ncells)
    end
    wpWeightIn .-= delta
    wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    return wpWeightIn, wpWeightOut
end
