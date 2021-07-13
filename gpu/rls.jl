function rls(p, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone)
    rtrim = reshape(r[Px], p.Lexc+p.Linh, 1, p.Ncells)
    k = batched_mul(P, rtrim)
    den = plusone ./ (plusone .+ batched_mul(batched_transpose(rtrim), k))
    CUBLAS.gemm_strided_batched!('N', 'T', minusone, k, batched_mul(k, den), plusone, P)
    e = batched_mul(batched_transpose(wpWeightIn), rtrim) .+
        reshape(synInputBalanced, 1,1,p.Ncells) .-
        reshape(xtarg[learn_seq,:], 1,1,p.Ncells)
    wpWeightIn .-= batched_mul(batched_mul(k, e), den)
    k = den = e = nothing
    learn_seq += 1
    wpWeightOut = convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,Array(wpWeightIn),wpWeightOut)
    return wpWeightIn, wpWeightOut, learn_seq
end
