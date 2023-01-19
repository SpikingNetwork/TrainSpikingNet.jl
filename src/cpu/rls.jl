function rls(itask,
             raug, k, delta, Ncells, Lei, r, s, Px, P, u_bal, utarg,
             learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero)

    @maybethread for ci = 1:Ncells
        raug_tid = @view raug[:,Threads.threadid()]
        raug_tid[1:Lei] = @views r[Px[ci]]
        raug_tid[Lei+1:end] = s
        k_tid = @view k[:,Threads.threadid()]

        @static if Param.PType == Array
            # k_tid .= P[ci] * raug / PScale
            # mul!(k_tid, P[ci], raug, 1.0/PScale, exactlyzero)
            gemv!('N', plusone/PScale, P[ci], raug_tid, exactlyzero, k_tid)
        elseif Param.PType == Symmetric
            symv!('U', plusone/PScale, P[ci].data, raug_tid, exactlyzero, k_tid)
        elseif Param.PType == SymmetricPacked
            spmv!('U', plusone/PScale, P[ci].tri, raug_tid, exactlyzero, k_tid)
        end

        vPv = raug_tid'*k_tid
        den = plusone/(plusone + vPv)

        @static if Param.PType == Array
            # P[ci] .-= den*k_tid*k_tid'
            # P[ci] .-= round.(PPrecision, clamp.(den*k_tid*k_tid' * PScale,
            #                                     typemin(PPrecision), typemax(PPrecision)))
            # mul!(P[ci], k_tid, transpose(k_tid), -den*PScale, plusone)
            ger!(-den*PScale, k_tid, k_tid, P[ci])
        elseif Param.PType == Symmetric
            syr!('U', -den*PScale, k_tid, P[ci].data)
        elseif Param.PType == SymmetricPacked
            spr!('U', -den*PScale, k_tid, P[ci].tri)
        end

        delta_tid = @view delta[:,Threads.threadid()]
        wpWeightInci = @view wpWeightIn[:,ci]
        e = wpWeightInci'*view(raug_tid, 1:Lei) + u_bal[ci] - utarg[learn_seq,ci,itask]
        @static if Param.LX>0
              wpWeightXci = @view wpWeightX[ci,:]
              e += wpWeightXci'*s
        end
        delta_tid = e.*k_tid.*den
        wpWeightInci .-= @view delta_tid[1:Lei]
        @static Param.LX>0 && (wpWeightXci .-= @view delta_tid[Lei+1:end])
    end
    wpWeightOut = convertWgtIn2Out(Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    return wpWeightIn, wpWeightOut
end
