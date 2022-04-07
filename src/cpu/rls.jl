function rls(k, Ncells, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, exactlyzero)

    @maybethread for ci = 1:Ncells
        rtrim = r[Px[ci]] 
        k_tid = @view k[:,Threads.threadid()]

        @static if p.PType == Array
            # k_tid .= P[ci] * rtrim / PScale
            # mul!(k_tid, P[ci], rtrim, 1.0/PScale, exactlyzero)
            BLAS.gemv!('N', plusone/PScale, P[ci], rtrim, exactlyzero, k_tid)
        elseif p.PType == Symmetric
            BLAS.symv!('U', plusone/PScale, P[ci].data, rtrim, exactlyzero, k_tid)
        elseif p.PType == SymmetricPacked
            BLAS.spmv!('U', plusone/PScale, P[ci].tri, rtrim, exactlyzero, k_tid)
        end

        vPv = rtrim'*k_tid
        den = plusone/(plusone + vPv)

        @static if p.PType == Array
            # P[ci] .-= den*k_tid*k_tid'
            # P[ci] .-= round.(PPrecision, clamp.(den*k_tid*k_tid' * PScale,
            #                                     typemin(PPrecision), typemax(PPrecision)))
            # mul!(P[ci], k_tid, transpose(k_tid), -den*PScale, plusone)
            BLAS.ger!(-den*PScale, k_tid, k_tid, P[ci])
        elseif p.PType == Symmetric
            BLAS.syr!('U', -den*PScale, k_tid, P[ci].data)
        elseif p.PType == SymmetricPacked
            SymmetricFormats.spr!('U', -den*PScale, k_tid, P[ci].tri)
        end

        wpWeightInci = @view wpWeightIn[:,ci]
        e = wpWeightInci'*rtrim + synInputBalanced[ci] - xtarg[learn_seq,ci]
        wpWeightInci .-= e.*k_tid.*den
    end
    wpWeightOut = convertWgtIn2Out(Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut
end
