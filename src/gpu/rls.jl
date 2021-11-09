function rls(k, den, e, delta, L, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone)
    rtrim = reshape(r[Px], 2*L, 1, Ncells)
    batched_mul!(k, P, rtrim)
    batched_mul!(den, batched_transpose(rtrim), k)
    den .= plusone ./ (plusone .+ den)
    delta .= k .* den
    batched_mul!(P, k, batched_transpose(delta), minusone, plusone)
    batched_mul!(e, batched_transpose(wpWeightIn), rtrim)
    e .+= reshape(synInputBalanced .- xtarg[learn_seq,:], 1,1,Ncells)
    delta .= k .* e
    delta .*= den
    wpWeightIn .-= delta
    wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    return wpWeightIn, wpWeightOut
end
