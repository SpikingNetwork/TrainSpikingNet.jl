function rls(raug, k, delta, Ncells, Lei, r, s, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightFfwd, wpWeightIn, wpWeightOut, plusone, exactlyzero)

    @maybethread for ci = 1:Ncells
        raug_tid = @view raug[:,Threads.threadid()]
        raug_tid[1:Lei] = @views r[Px[ci]]
        raug_tid[Lei+1:end] = s
        k_tid = @view k[:,Threads.threadid()]

        @static if p.PType == Array
            # k_tid .= P[ci] * raug / PScale
            # mul!(k_tid, P[ci], raug, 1.0/PScale, exactlyzero)
            BLAS.gemv!('N', plusone/PScale, P[ci], raug_tid, exactlyzero, k_tid)
        elseif p.PType == Symmetric
            BLAS.symv!('U', plusone/PScale, P[ci].data, raug_tid, exactlyzero, k_tid)
        elseif p.PType == SymmetricPacked
            BLAS.spmv!('U', plusone/PScale, P[ci].tri, raug_tid, exactlyzero, k_tid)
        end

        vPv = raug_tid'*k_tid
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

        delta_tid = @view delta[:,Threads.threadid()]
        wpWeightInci = @view wpWeightIn[:,ci]
        e = wpWeightInci'*view(raug_tid, 1:Lei) + synInputBalanced[ci] - xtarg[learn_seq,ci]
        @static if p.Lffwd>0
              wpWeightFfwdci = @view wpWeightFfwd[ci,:]
              e += wpWeightFfwdci'*s
        end
        delta_tid = e.*k_tid.*den
        wpWeightInci .-= @view delta_tid[1:Lei]
        @static p.Lffwd>0 && (wpWeightFfwdci .-= @view delta_tid[Lei+1:end])
    end
    wpWeightOut = convertWgtIn2Out(Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut
end
