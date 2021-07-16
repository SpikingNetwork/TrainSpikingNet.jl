function rls(k, p, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone)

    for ci = 1:p.Ncells
        rtrim = r[Px[ci]] 
        mul!(k, P[ci], rtrim)
        vPv = rtrim'*k
        den = plusone/(plusone + vPv)
        BLAS.gemm!('N','T',minusone,k,k*den,plusone,P[ci])

        e  = wpWeightIn[ci,:]'*rtrim + synInputBalanced[ci] - xtarg[learn_seq,ci]
        wpWeightIn[ci,:] .-= e*k*den
    end
    learn_seq += 1
    wpWeightOut = convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut, learn_seq
end
