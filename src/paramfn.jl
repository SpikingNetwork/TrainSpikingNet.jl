module Param
    using LinearAlgebra, SymmetricFormats, Random, JLD2, UnPack

    get_param() = (;
            PPrecision, PScale, FloatPrecision, IntPrecision, PType, PCompute,
            seed, rng_func, rng,
            example_neurons, wid, maxrate,
            penlambda, penlamFF, penmu,
            learn_every, PHistory,
            train_duration, stim_on, stim_off, train_time,
            dt, Nsteps, u0_skip_time, u0_ncells,
            Ncells, Ne, Ni,
            tau_meme, tau_memi, g,
            K, LX,
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
            benchmark,
            )
    get_init_code() = init_code
end

function param(data_dir)
    Param.include(joinpath(data_dir, "param.jl"))
    p = Param.get_param()
    save(joinpath(data_dir, "param.jld2"), "param", p, "init_code", Param.get_init_code())

    if p.Ncells == typemax(p.IntPrecision)
        @warn "IntPrecision is too small for GPU (but fine for CPU)"
    elseif p.Ncells > typemax(p.IntPrecision)
        @error "IntPrecision is too small"
    end

    if p.PCompute == :small && p.PType == SymmetricPacked
       @error "PType must be Array or Symmetric if PCompute is :small"
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
    global init_code = load(joinpath(_data_dir, "param.jld2"), "init_code")
    eval(init_code)
    global p = load(joinpath(_data_dir, "param.jld2"), "param")
    global choose_task = eval(p.choose_task_func)

    pu==:gpu && p.PCompute==:small && p.PType==SymmetricPacked &&
            @error "For :gpu, PType can only be Array or Symmetric if PCompute is :small"

    global extra = pu==:cpu ? 0 : 1

    # --- load code --- #
    include(p.genStaticWeights_file)
    include(p.genPlasticWeights_file)
    include(p.genRateX_file)
    include(p.genUTarget_file)
    include(p.genXStim_file)
    include(p.cellModel_file)
    include(joinpath(@__DIR__, "scratch.jl"))
    include(joinpath(@__DIR__, pu_str, "variables.jl"))
    include(joinpath(@__DIR__, pu_str, "loop.jl"))
    include(joinpath(@__DIR__, pu_str, "rls-$(p.PCompute).jl"))
    include(joinpath(@__DIR__, pu_str, "wpWeightIn2Out.jl"))
    include(joinpath(@__DIR__, pu_str, "trainfn.jl"))
    include(joinpath(@__DIR__, pu_str, "testfn.jl"))

    nothing
end
