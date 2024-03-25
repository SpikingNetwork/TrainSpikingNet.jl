function rls(itask,
             raug::Matrix{T}, k, rrXg, delta, Ncells, r, rX, Pinv, u_bal, utarg,
             rrXhistory, charge0, LX, penmu, penlamFF, penlambda,
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
        Pinv_tid = @view Pinv[1:lenrX+ncpIn, 1:lenrX+ncpIn, Threads.threadid()]
        rrXg_tid = @view rrXg[1:lenrX+ncpIn, Threads.threadid()]

        generate_Pinv!(Pinv_tid, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)

        for rrXi in eachcol(rrXhistory)
            copyto!(rrXg_tid, 1, rrXi, 1, lenrX)
            for i=1:ncpIn
                rrXg_tid[lenrX+i] = rrXi[wpIndexIn[ci][i]]
            end
            @static if p.PType == Array
                ger!(plusone, rrXg_tid, rrXg_tid, Pinv_tid)
            elseif p.PType == Symmetric
                syr!('U', plusone, rrXg_tid, Pinv_tid)
            end
        end

        @static if p.PType == Array
            ldiv!(k_tid, lu!(Pinv_tid), _raug_tid / PScale)
        elseif p.PType == Symmetric
            ldiv!(k_tid, bunchkaufman!(Symmetric(Pinv_tid)), _raug_tid / PScale)
        end
        
        delta_tid = @view delta[1:lenrX+ncpIn, Threads.threadid()]
        e = wpWeightIn[ci]' * (@view raug_tid[lenrX+1:end]) + u_bal[ci] - utarg[learn_seq,ci,itask]
        @static p.LX>0 && (e += view(wpWeightX,ci,:)' * rX)
        delta_tid .= e .* k_tid
        wpWeightIn[ci] .-= @view delta_tid[lenrX+1:end]
        @static p.LX>0 && (wpWeightX[ci,:] .-= @view delta_tid[1:lenrX])
    end
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    push!(rrXhistory, 0)
    @static p.LX>0 && (rrXhistory[1:lenrX, end] = rX)
    rrXhistory[lenrX+1:end, end] = r

    return wpWeightIn, wpWeightOut
end
