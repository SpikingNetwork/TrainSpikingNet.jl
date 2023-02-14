function test(; ntrials = 1,
                ineurons_to_test = 1:16,
                restore_from_checkpoint = nothing,
                no_plot = false)

    Threads.nthreads() < ndevices() && @warn "performance is best if no. threads is set to no. GPUs"
    if Threads.nthreads() > ndevices()
        @error "no. threads cannot exceed no. GPUs"
        exit()
    end

    # --- load initialization --- #
    nc0 = load(joinpath(data_dir,"nc0.jld2"), "nc0")
    X_stim = load(joinpath(data_dir,"X_stim.jld2"), "X_stim")
    w0Index = load(joinpath(data_dir,"w0Index.jld2"), "w0Index")
    w0Weights = load(joinpath(data_dir,"w0Weights.jld2"), "w0Weights")
    wpIndexIn = load(joinpath(data_dir,"wpIndexIn.jld2"), "wpIndexIn")
    wpIndexOut = load(joinpath(data_dir,"wpIndexOut.jld2"), "wpIndexOut")
    wpIndexConvert = load(joinpath(data_dir,"wpIndexConvert.jld2"), "wpIndexConvert")
    if isnothing(restore_from_checkpoint)
        R = maximum([parse(Int, m.captures[1])
                     for m in match.(r"ckpt([0-9]+)\.jld2",
                                     filter(startswith("wpWeightIn-ckpt"),
                                            readdir(data_dir)))])
    else
        R = restore_from_checkpoint
    end
    wpWeightX = load(joinpath(data_dir,"wpWeightX.jld2"), "wpWeightX")
    wpWeightIn = load(joinpath(data_dir,"wpWeightIn-ckpt$R.jld2"))["wpWeightIn"]
    wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells)
    wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    # --- set up variables --- #
    global nc0 = CuArray{p.IntPrecision}(nc0)
    global X_stim = CuArray{p.FloatPrecision}(X_stim);
    global w0Index = CuArray{p.IntPrecision}(w0Index);
    global w0Weights = CuArray{p.FloatPrecision}(w0Weights);
    global wpIndexOut = CuArray{p.IntPrecision}(wpIndexOut);
    global wpWeightOut = CuArray{p.FloatPrecision}(wpWeightOut);
    global wpWeightX = CuArray{p.FloatPrecision}(wpWeightX);

    rng = eval(p.rng_func.gpu)
    isnothing(p.seed) || Random.seed!(rng, p.seed)
    save(joinpath(data_dir,"rng-test.jld2"), "rng", rng)

    # --- test the network --- #
    ntasks = size(X_stim,3)
    nss = Array{Any}(undef, ntrials, ntasks);
    timess = Array{Any}(undef, ntrials, ntasks);
    utotals = Array{Any}(undef, ntrials, ntasks);
    copy_rng = [typeof(rng)() for _=1:ndevices()];
    isnothing(p.seed) || Random.seed!.(copy_rng, p.seed)
    for var in [:times, :ns, :timesX, :nsX, :X_stim, :nc0,
                :w0Index, :w0Weights, :wpWeightX, :wpIndexOut, :wpWeightOut, :X_bal,
                :inputsE, :inputsI, :inputsP, :inputsEPrev, :inputsIPrev, :inputsPPrev,
                :u_bale, :u_bali, :uX_plas, :u_bal, :u,
                :X, :lastSpike, :bnotrefrac, :bspike, :v, :noise, :sig]
      @eval (device!(0); tmp = Array($var))
      @eval $(Symbol("copy_",var)) = [(device!(idevice-1); CuArray(tmp)) for idevice=1:ndevices()];
    end
    if typeof(p.tau_plas)<:AbstractArray
        device!(0); tmp = Array(invtau_plas)
        copy_invtau_plas = [(device!(idevice-1); CuArray(tmp)) for idevice=1:ndevices()];
    end
    synchronize()
    Threads.@threads for itrial=1:ntrials
        idevice = Threads.threadid()
        device!(idevice-1)
        for itask = 1:ntasks
            t = @elapsed thisns, thistimes, _, _, thisutotal, _ = loop_test(itask,
                  p.learn_every, p.stim_on, p.stim_off,
                  p.train_time, p.dt, p.Nsteps, p.Ncells,
                  nothing, nothing, p.LX, p.refrac, learn_step, invtau_bale,
                  invtau_bali,
                  typeof(p.tau_plas)<:Number ? invtau_plas : copy_invtau_plas[idevice],
                  copy_X_bal[idevice],
                  maxTimes,
                  copy_times[idevice],
                  copy_ns[idevice],
                  copy_timesX[idevice],
                  copy_nsX[idevice],
                  copy_inputsE[idevice],
                  copy_inputsI[idevice],
                  copy_inputsP[idevice],
                  copy_inputsEPrev[idevice],
                  copy_inputsIPrev[idevice],
                  copy_inputsPPrev[idevice],
                  nothing, nothing, nothing, nothing,
                  copy_u_bale[idevice],
                  copy_u_bali[idevice],
                  copy_uX_plas[idevice],
                  copy_u_bal[idevice],
                  copy_u[idevice],
                  nothing, nothing,
                  copy_X[idevice],
                  p.wid, p.example_neurons,
                  copy_lastSpike[idevice],
                  copy_bnotrefrac[idevice],
                  copy_bspike[idevice],
                  plusone,
                  nothing, nothing, nothing, nothing, nothing, nothing, nothing,
                  copy_v[idevice],
                  copy_rng[idevice],
                  copy_noise[idevice],
                  nothing,
                  copy_sig[idevice],
                  nothing,
                  copy_w0Index[idevice],
                  copy_w0Weights[idevice],
                  copy_nc0[idevice],
                  copy_X_stim[idevice],
                  nothing,
                  copy_wpWeightX[idevice],
                  nothing,
                  copy_wpIndexOut[idevice],
                  nothing, nothing,
                  copy_wpWeightOut[idevice],
                  nothing,
                  cellModel_args);
            nss[itrial, itask] = Array(thisns[ineurons_to_test])
            timess[itrial, itask] = Array(thistimes[ineurons_to_test,:])
            utotals[itrial, itask] = Array(thisutotal[:,ineurons_to_test])
            println("trial #", itrial, ", task #", itask, ": ",round(t, sigdigits=3), " sec")
        end
    end

    save(joinpath(data_dir, "test.jld2"),
         "ineurons_to_test", ineurons_to_test,
         "nss", nss, "timess", timess, "utotals", utotals)

    no_plot || run(`$(Base.julia_cmd())
                    $(joinpath(@__DIR__, "..", "plot.jl"))
                    -i $(repr(ineurons_to_test))
                    $(joinpath(data_dir, "test.jld2"))`);

    return (; nss, timess, utotals)
end
