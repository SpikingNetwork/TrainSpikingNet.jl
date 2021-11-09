function rls(k, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, exactlyzero)

    @maybethread for ci = 1:Ncells
        rtrim = r[Px[ci]] 
        @static if p.PType == Array
            # k[:,ci] .= P[ci] * rtrim
            # mul!(@view k[:,ci], P[ci], rtrim)
            BLAS.gemv!('N', plusone, P[ci], rtrim, exactlyzero, @view k[:,ci])
        elseif p.PType == Symmetric
            BLAS.symv!('U', plusone, P[ci].data, rtrim, exactlyzero, @view k[:,ci])
        elseif p.PType == SymmetricPacked
            BLAS.spmv!('U', plusone, P[ci].tri, rtrim, exactlyzero, @view k[:,ci])
        end
        vPv = rtrim'*k[:,ci]
        den = plusone/(plusone + vPv)
        @static if p.PType == Array
            # P[ci] .-= den*k[:,ci]*k[:,ci]'
            # mul!(P[ci], k[:,ci], transpose(k[:,ci]), -den, plusone)
            BLAS.ger!(-den, k[:,ci], k[:,ci], P[ci])
        elseif p.PType == Symmetric
            BLAS.syr!('U', -den, k[:,ci], P[ci].data)
        elseif p.PType == SymmetricPacked
            PackedArrays.spr!('U', -den, k[:,ci], P[ci].tri)
        end
        e  = wpWeightIn[ci,:]'*rtrim + synInputBalanced[ci] - xtarg[learn_seq,ci]
        wpWeightIn[ci,:] .-= e*k[:,ci]*den
    end
    wpWeightOut = convertWgtIn2Out(Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut
end
