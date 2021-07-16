function rls(k, den, e, L, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone)
    rtrim = reshape(r[Px], 2*L, 1, Ncells)
    NNlib.batched_mul!(k, P, rtrim)
    NNlib.batched_mul!(den, batched_transpose(rtrim), k)
    den .= plusone ./ (plusone .+ den)
    CUBLAS.gemm_strided_batched!('N', 'T', minusone, k, batched_mul(k, den), plusone, P)
    NNlib.batched_mul!(e, batched_transpose(wpWeightIn), rtrim)
    e .+= reshape(synInputBalanced .- xtarg[learn_seq,:], 1,1,Ncells)
    wpWeightIn .-= batched_mul(batched_mul(k, e), den)
    learn_seq += 1
    wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    return wpWeightIn, wpWeightOut, learn_seq
end
