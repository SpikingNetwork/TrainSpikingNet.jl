function rls(k, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone)

    @maybethread for ci = 1:Ncells
        rtrim = r[Px[ci]] 
        k[:,ci] .= P[ci] * rtrim
        vPv = rtrim'*k[:,ci]
        den = plusone/(plusone + vPv)
        mul!(P[ci], k[:,ci], transpose(k[:,ci]), -den, plusone)
        e  = wpWeightIn[ci,:]'*rtrim + synInputBalanced[ci] - xtarg[learn_seq,ci]
        wpWeightIn[ci,:] .-= e*k[:,ci]*den
    end
    wpWeightOut = convertWgtIn2Out(Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut
end
