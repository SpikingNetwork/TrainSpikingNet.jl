function rls(itask,
             raug::Matrix{T}, k, rrXg, delta, Ncells, r, rX, Pinv, work, lwork, pivot,
             u_bal, utarg, rrXhistory, charge0, LX, penmu, penlamFF, penlambda,
             learn_seq, wpIndexIn, wpIndexConvert, wpWeightX, wpWeightIn,
             wpWeightOut, plusone, exactlyzero, PScale) where T

    @maybethread :static for ci = 1:Ncells
        ncpIn = length(wpIndexIn[ci])
        raug_tid = @view raug[1:LX+ncpIn, Threads.threadid()]
        @static p.LX>0 && (raug_tid[1:LX] = rX)
        raug_tid[LX+1:end] = @views r[wpIndexIn[ci]]
        _raug_tid = T<:Real ? raug_tid : ustrip(raug_tid)
        k_tid = @view k[1:LX+ncpIn, Threads.threadid()]
        Pinv_tid = @view Pinv[1:LX+ncpIn, 1:LX+ncpIn, Threads.threadid()]
        @static p.PType == Symmetric && (work_tid = work[Threads.threadid()])
        pivot_tid = pivot[Threads.threadid()]
        rrXg_tid = @view rrXg[1:LX+ncpIn, Threads.threadid()]

        generate_Pinv!(Pinv_tid, ci, wpWeightIn, charge0, LX, penmu, penlamFF, penlambda)

        for rrXi in eachcol(rrXhistory)
            copyto!(rrXg_tid, 1, rrXi, 1, LX)
            for i=1:ncpIn
                rrXg_tid[LX+i] = rrXi[wpIndexIn[ci][i]]
            end
            @static if p.PType == Array
                ger!(plusone, rrXg_tid, rrXg_tid, Pinv_tid)
            elseif p.PType == Symmetric
                syr!('U', plusone, rrXg_tid, Pinv_tid)
            end
        end

        _raug_tid ./= PScale
        @static if p.PType == Array
            _f = lu!(Pinv_tid, pivot_tid)
            ldiv!(k_tid, _f, _raug_tid)
        elseif p.PType == Symmetric
            _f = bunchkaufman!(Symmetric(Pinv_tid), work_tid, lwork, pivot_tid)
            ldiv!(k_tid, _f, _raug_tid)
        end
        
        delta_tid = @view delta[1:LX+ncpIn, Threads.threadid()]
        e = wpWeightIn[ci]' * (@view raug_tid[LX+1:end]) + u_bal[ci] - utarg[learn_seq,ci,itask]
        @static p.LX>0 && (e += view(wpWeightX,ci,:)' * rX)
        delta_tid .= e .* k_tid
        @static if !p.benchmark
            wpWeightIn[ci] .-= @view delta_tid[LX+1:end]
            @static p.LX>0 && (wpWeightX[ci,:] .-= @view delta_tid[1:LX])
        end
    end
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    push!(rrXhistory, 0)
    @static p.LX>0 && (rrXhistory[1:LX, end] = rX)
    rrXhistory[LX+1:end, end] = r

    return wpWeightIn, wpWeightOut
end


LinearAlgebra.lu!(A::StridedMatrix{<:LinearAlgebra.BLAS.BlasFloat}, ipiv;
                  check::Bool = true, allowsingular::Bool = false) =
        LinearAlgebra.lu!(A, ipiv, RowMaximum(); check, allowsingular)
function LinearAlgebra.lu!(A::StridedMatrix{T}, ipiv, ::RowMaximum;
                           check::Bool = true, allowsingular::Bool = false) where
                           {T<:LinearAlgebra.BLAS.BlasFloat}
    lpt = LinearAlgebra.LAPACK.getrf!(A, ipiv; check)
    #check && LinearAlgebra._check_lu_success(lpt[3], allowsingular)
    check && LinearAlgebra.checknonsingular(lpt[3])
    return LinearAlgebra.LU{T,typeof(lpt[1]),typeof(lpt[2])}(lpt[1], lpt[2], lpt[3])
end

if VERSION < v"1.11"
    for (getrf, elty) in ((:sgetrf_,:Float32), (:dgetrf_,:Float64))
        @eval function LinearAlgebra.LAPACK.getrf!(A::AbstractMatrix{$elty},
                                                   ipiv::AbstractVector{LinearAlgebra.BLAS.BlasInt};
                                                   check::Bool=true)
            LinearAlgebra.BLAS.require_one_based_indexing(A)
            check && LinearAlgebra.LAPACK.chkfinite(A)
            LinearAlgebra.chkstride1(A)
            m, n = size(A)
            lda  = max(1,stride(A, 2))
            info = Ref{LinearAlgebra.BLAS.BlasInt}()
            ccall((LinearAlgebra.BLAS.@blasfunc($getrf), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                  (Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{$elty},
                   Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{LinearAlgebra.BLAS.BlasInt},
                   Ptr{LinearAlgebra.BLAS.BlasInt}),
                  m, n, A, lda, ipiv, info)
            LinearAlgebra.LAPACK.chkargsok(info[])
            A, ipiv, info[] #Error code is stored in LU factorization type
        end
    end
end


function get_workspace(::Type{LinearAlgebra.BunchKaufman},
                       A::LinearAlgebra.RealHermSymComplexSym{<:LinearAlgebra.BLAS.BlasReal,<:StridedMatrix},
                                     rook::Bool = false)
    _work, _lwork = Vector{eltype(A)}(undef, 1), LinearAlgebra.BLAS.BlasInt(-1)
    work, lwork, ipiv = rook ? LinearAlgebra.LAPACK.sytrf_rook!(A.uplo, A.data, _work, _lwork) :
                        LinearAlgebra.LAPACK.sytrf!(A.uplo, A.data, _work, _lwork)
    return work, lwork, ipiv
end

function LinearAlgebra.bunchkaufman!(A::LinearAlgebra.RealHermSymComplexSym{<:LinearAlgebra.BLAS.BlasReal,<:StridedMatrix},
                                     rook::Bool = false; check::Bool = true)
    work, lwork, ipiv = LinearAlgebra.get_workspace(LinearAlgebra.BunchKaufman, A, rook)
    bunchkaufman!(A, work, lwork, ipiv, rook, check)
end

function LinearAlgebra.bunchkaufman!(A::LinearAlgebra.RealHermSymComplexSym{<:LinearAlgebra.BLAS.BlasReal,<:StridedMatrix},
                                     work, lwork, ipiv, rook::Bool = false; check::Bool = true)
    LD, ipiv, info = rook ? LinearAlgebra.LAPACK.sytrf_rook!(A.uplo, A.data, work, lwork, ipiv) :
                            LinearAlgebra.LAPACK.sytrf!(A.uplo, A.data, work, lwork, ipiv)
    check && LinearAlgebra.checknonsingular(info)
    LinearAlgebra.BunchKaufman(LD, ipiv, A.uplo, true, rook, info)
end

for (sytrf, elty) in ((:dsytrf_,:Float64), (:ssytrf_,:Float32))
    """
    allocate `work` and use it to compute and return `LDLT`
    """
    @eval function LinearAlgebra.LAPACK.sytrf!(uplo::AbstractChar, A::AbstractMatrix{$elty})
        work, lwork = Vector{$elty}(undef, 1), LinearAlgebra.BLAS.BlasInt(-1)
        _, _, ipiv = LinearAlgebra.LAPACK.sytrf!(uplo, A, work, lwork)
        LinearAlgebra.LAPACK.sytrf!(uplo, A, work, lwork, ipiv)
    end

    """
    if `lwork == -1`, only allocate and return `work`.  otherwise use `work` to
    compute and return `LDLT`
    """
    @eval function LinearAlgebra.LAPACK.sytrf!(uplo::AbstractChar, A::AbstractMatrix{$elty},
                                               work, lwork, _ipiv::T=nothing) where
                T<:Union{Nothing,AbstractVector{LinearAlgebra.BLAS.BlasInt}}
        lwork0 = lwork
        LinearAlgebra.chkstride1(A)
        n = LinearAlgebra.checksquare(A)
        LinearAlgebra.BLAS.chkuplo(uplo)
        ipiv = T<:Nothing ? similar(A, LinearAlgebra.BLAS.BlasInt, n) : _ipiv
        if n == 0
            if lwork0 == -1
                return work, lwork, ipiv
            else
                return A, ipiv, zero(LinearAlgebra.BLAS.BlasInt)
            end
        end
        info  = Ref{LinearAlgebra.BLAS.BlasInt}()
        ccall((LinearAlgebra.BLAS.@blasfunc($sytrf), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
              (Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{$elty},
               Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{LinearAlgebra.BLAS.BlasInt}, Ptr{$elty},
               Ref{LinearAlgebra.BLAS.BlasInt}, Ptr{LinearAlgebra.BLAS.BlasInt}, Clong),
              uplo, n, A, stride(A,2), ipiv, work, lwork, info, 1)
        LinearAlgebra.LAPACK.chkargsok(info[])
        if lwork0 == -1
            lwork = LinearAlgebra.BLAS.BlasInt(real(work[1]))
            resize!(work, lwork)
            return work, lwork, ipiv
        else
            return A, ipiv, info[]
        end
    end
end
