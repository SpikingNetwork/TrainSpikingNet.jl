function rls(itask,
             raug::Matrix{T}, k, delta, Ncells, r, rX, P, u_bal, utarg,
             learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero, PScale) where T

    lenrX = length(rX)
    @maybethread :static for ci = 1:Ncells
        ncpIn = length(wpIndexIn[ci])
        raug_tid = @view raug[1:lenrX+ncpIn, Threads.threadid()]
        @static p.LX>0 && (raug_tid[1:lenrX] = rX)
        raug_tid[lenrX+1:end] = @views r[wpIndexIn[ci]]
        _raug_tid = T<:Real ? raug_tid : ustrip(raug_tid)
        k_tid = @view k[1:lenrX+ncpIn, Threads.threadid()]

        @static if p.PType == Array
            # k_tid .= P[ci] * _raug_tid / PScale
            # mul!(k_tid, P[ci], _raug_tid, 1.0/PScale, exactlyzero)
            gemv!('N', plusone/PScale, P[ci], _raug_tid, exactlyzero, k_tid)
        elseif p.PType == Symmetric
            symv!('U', plusone/PScale, P[ci].data, _raug_tid, exactlyzero, k_tid)
        elseif p.PType == SymmetricPacked
            spmv!('U', plusone/PScale, P[ci].tri, _raug_tid, exactlyzero, k_tid)
        end

        vPv = raug_tid' * k_tid
        if T<:Real
            den = plusone / (plusone + vPv)
            _den = den
        else
            den = plusone / (oneunit(vPv) + vPv)
            _den = ustrip(den)
        end

        @static if p.PType == Array
            # P[ci] .-= den * PScale * k_tid * k_tid'
            # P[ci] .-= round.(PPrecision, clamp.(_den*k_tid*k_tid' * PScale,
            #                                     typemin(PPrecision), typemax(PPrecision)))
            # mul!(P[ci], k_tid, transpose(k_tid), -_den*PScale, plusone)
            ger!(-_den*PScale, k_tid, k_tid, P[ci])
        elseif p.PType == Symmetric
            syr!('U', -_den*PScale, k_tid, P[ci].data)
        elseif p.PType == SymmetricPacked
            spr!('U', -_den*PScale, k_tid, P[ci].tri)
        end

        delta_tid = @view delta[1:lenrX+ncpIn, Threads.threadid()]
        e = wpWeightIn[ci]' * (@view raug_tid[lenrX+1:end]) + u_bal[ci] - utarg[learn_seq,ci,itask]
        @static p.LX>0 && (e += view(wpWeightX,ci,:)' * rX)
        delta_tid .= e .* k_tid
        @static if !p.benchmark
            wpWeightIn[ci] .-= @view delta_tid[lenrX+1:end]
            @static p.LX>0 && (wpWeightX[ci,:] .-= @view delta_tid[1:lenrX])
        end
    end
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    return wpWeightIn, wpWeightOut
end
