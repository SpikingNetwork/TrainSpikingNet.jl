function rls(itask,
             raug, k, delta, Ncells, r, rX, P, u_bal, utarg,
             learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero, PScale)

    lenrX = length(rX)
    @maybethread for ci = 1:Ncells
        ncpIn = length(wpIndexIn[ci])
        raug_tid = @view raug[1:lenrX+ncpIn, Threads.threadid()]
        @static p.LX>0 && (raug_tid[1:lenrX] = rX)
        raug_tid[lenrX+1:end] = @views r[wpIndexIn[ci]]
        k_tid = @view k[1:lenrX+ncpIn, Threads.threadid()]

        @static if p.PType == Array
            # k_tid .= P[ci] * raug_tid / PScale
            # mul!(k_tid, P[ci], raug, 1.0/PScale, exactlyzero)
            gemv!('N', plusone/PScale, P[ci], raug_tid, exactlyzero, k_tid)
        elseif p.PType == Symmetric
            symv!('U', plusone/PScale, P[ci].data, raug_tid, exactlyzero, k_tid)
        elseif p.PType == SymmetricPacked
            spmv!('U', plusone/PScale, P[ci].tri, raug_tid, exactlyzero, k_tid)
        end

        vPv = raug_tid'*k_tid
        den = plusone/(plusone + vPv)

        @static if p.PType == Array
            # P[ci] .-= den * PScale * k_tid * k_tid'
            # P[ci] .-= round.(PPrecision, clamp.(den*k_tid*k_tid' * PScale,
            #                                     typemin(PPrecision), typemax(PPrecision)))
            # mul!(P[ci], k_tid, transpose(k_tid), -den*PScale, plusone)
            ger!(-den*PScale, k_tid, k_tid, P[ci])
        elseif p.PType == Symmetric
            syr!('U', -den*PScale, k_tid, P[ci].data)
        elseif p.PType == SymmetricPacked
            spr!('U', -den*PScale, k_tid, P[ci].tri)
        end

        delta_tid = @view delta[1:lenrX+ncpIn, Threads.threadid()]
        e = wpWeightIn[ci]' * (@view raug_tid[lenrX+1:end]) + u_bal[ci] - utarg[learn_seq,ci,itask]
        @static if p.LX>0
            e += view(wpWeightX,ci,:)' * rX
        end
        delta_tid = e .* k_tid .* den
        wpWeightIn[ci] .-= @view delta_tid[lenrX+1:end]
        @static p.LX>0 && (wpWeightX[ci,:] .-= @view delta_tid[1:lenrX])
    end
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    return wpWeightIn, wpWeightOut
end
