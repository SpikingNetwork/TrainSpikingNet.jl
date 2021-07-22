function rls(k, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone)

    Threads.@threads for ci = 1:Ncells
        rtrim = r[Px[ci]] 
        k[:,ci] .= P[ci] * rtrim
        vPv = rtrim'*k[:,ci]
        den = plusone/(plusone + vPv)
        BLAS.gemm!('N','T',-den,k[:,ci],k[:,ci],plusone,P[ci])

        e  = wpWeightIn[ci,:]'*rtrim + synInputBalanced[ci] - xtarg[learn_seq,ci]
        wpWeightIn[ci,:] .-= e*k[:,ci]*den
    end
    wpWeightOut = convertWgtIn2Out(Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut
end
