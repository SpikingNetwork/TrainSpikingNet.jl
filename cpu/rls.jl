function rls(p, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut)

    for ci = 1:p.Ncells
        rtrim = @view r[Px[ci]] 
        k = P[ci]*rtrim
        vPv = rtrim'*k
        den = 1.0/(1.0 + vPv[1])
        BLAS.gemm!('N','T',-1.0,k,k*den,1.0,P[ci])

        e  = wpWeightIn[ci,:]'*rtrim + synInputBalanced[ci] - xtarg[learn_seq,ci]
        dw = -e*k*den
        wpWeightIn[ci,:] .+= dw      
    end
    learn_seq += 1
    wpWeightOut = convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut, learn_seq
end
