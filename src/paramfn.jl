module Param
    using LinearAlgebra, SymmetricFormats, Random, JLD2, UnPack

    get_param() = (;
            PPrecision, PScale, FloatPrecision, IntPrecision, PType,
            seed, rng_func, rng,
            example_neurons, wid, maxrate,
            penlambda, penlamFF, penmu,
            learn_every,
            train_duration, stim_on, stim_off, train_time,
            dt, Nsteps,
            Ncells, Ne, Ni,
            tau_meme, tau_memi,
            K, L, LX, Lexc, Linh,
            X_bal,
            vre, threshe, threshi, refrac,
            tau_bale, tau_bali, tau_plas,
            noise_model, sig,
            correlation_var,
            :genXStim_file => abspath(@__DIR__, genXStim_file),
            genXStim_args,
            :genUTarget_file => abspath(@__DIR__, genUTarget_file),
            genUTarget_args,
            :genRateX_file => abspath(@__DIR__, genRateX_file),
            genRateX_args,
            :genStaticWeights_file => abspath(@__DIR__, genStaticWeights_file),
            genStaticWeights_args,
            :genPlasticWeights_file => abspath(@__DIR__, genPlasticWeights_file),
            genPlasticWeights_args,
            :cellModel_file => abspath(@__DIR__, cellModel_file),
            cellModel_args,
            choose_task_func,
            )
end

function param(data_dir)
    Param.include(joinpath(data_dir, "param.jl"))
    p = Param.get_param()
    save(joinpath(data_dir, "param.jld2"), "param", p)

    if p.Ncells == typemax(p.IntPrecision)
      @warn "IntPrecision is too small for GPU (but fine for CPU)"
    elseif p.Ncells > typemax(p.IntPrecision)
      @error "IntPrecision is too small"
    end

    return p
end

function config(_data_dir, pu::Symbol=:cpu)
    @assert pu in [:cpu, :gpu]
    pu_str = string(pu)

    if pu==:cpu
        println(BLAS.get_config())
        BLAS.set_num_threads(1)
        Threads.nthreads()==1 && @warn "running single threaded"
    end

    global data_dir = _data_dir
    global p = load(joinpath(_data_dir, "param.jld2"), "param")
    
    # --- load code --- #
    include(p.genStaticWeights_file)
    include(p.genPlasticWeights_file)
    include(p.genRateX_file)
    include(p.genUTarget_file)
    include(p.genXStim_file)
    include(p.cellModel_file)

    include(joinpath(@__DIR__, pu_str, "variables.jl"))

    global choose_task = eval(p.choose_task_func)

    global kind

    kind = :init
    include(joinpath(@__DIR__, pu_str, "loop.jl"))
    include(joinpath(@__DIR__, "rate2utarg.jl"))

    kind = :train
    include(joinpath(@__DIR__, pu_str, "convertWgtIn2Out.jl"))
    include(joinpath(@__DIR__, pu_str, "loop.jl"))
    include(joinpath(@__DIR__, pu_str, "rls.jl"))

    kind = :train_test
    include(joinpath(@__DIR__, pu_str, "loop.jl"))

    kind = :test
    include(joinpath(@__DIR__, pu_str, "convertWgtIn2Out.jl"))
    include(joinpath(@__DIR__, pu_str, "loop.jl"))
    include(joinpath(@__DIR__, pu_str, "rls.jl"))

    include(joinpath(@__DIR__, pu_str, "trainfn.jl"))
    include(joinpath(@__DIR__, pu_str, "testfn.jl"))

    nothing
end
