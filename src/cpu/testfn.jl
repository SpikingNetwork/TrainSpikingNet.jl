function test(; ntrials = 1,
                ineurons_to_test = 1:16,
                restore_from_checkpoint = nothing,
                no_plot = false)

    # --- load initialization --- #
    nc0 = load(joinpath(data_dir,"nc0.jld2"), "nc0")
    ncpIn = load(joinpath(data_dir,"ncpIn.jld2"), "ncpIn")
    ncpOut = load(joinpath(data_dir,"ncpOut.jld2"), "ncpOut")
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
    wpWeightX = load(joinpath(data_dir,"wpWeightX-ckpt$R.jld2"), "wpWeightX");
    wpWeightIn = load(joinpath(data_dir,"wpWeightIn-ckpt$R.jld2"), "wpWeightIn")
    wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells)
    wpWeightOut = convertWgtIn2Out(p.Ncells, ncpIn,
                                   wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut)

    # --- set up variables --- #
    nc0 = Array{p.IntPrecision}(nc0)
    ncpOut = Array{p.IntPrecision}(ncpOut);
    X_stim = Array{p.FloatPrecision}(X_stim);
    w0Index = Array{p.IntPrecision}(w0Index);
    w0Weights = Array{p.FloatPrecision}(w0Weights);
    wpIndexOut = Array{p.IntPrecision}(wpIndexOut);
    wpWeightOut = Array{p.FloatPrecision}(wpWeightOut);
    wpWeightX = Array{p.FloatPrecision}(wpWeightX);

    rng = eval(p.rng_func.cpu)

    # --- test the network --- #
    ntasks = size(X_stim,3)
    nss = Array{Any}(undef, ntrials, ntasks);
    timess = Array{Any}(undef, ntrials, ntasks);
    utotals = Array{Any}(undef, ntrials, ntasks);
    copy_rng = [typeof(rng)() for _=1:Threads.nthreads()];
    isnothing(p.seed) || Random.seed!.(copy_rng, p.seed .+ (1:Threads.nthreads()))
    save(joinpath(data_dir,"rng-test.jld2"), "rng", copy_rng)
    for var in [:times, :ns, :timesX, :nsX,
                :inputsE, :inputsI, :inputsP, :inputsEPrev, :inputsIPrev, :inputsPPrev,
                :u_bale, :u_bali, :uX_plas, :u_bal, :u,
                :X, :lastSpike, :v, :noise]
        @eval $(Symbol("copy_",var)) = [deepcopy($var) for _=1:Threads.nthreads()];
    end
    Threads.@threads for itrial=1:ntrials
        for itask = 1:ntasks
            t = @elapsed thisns, thistimes, _, _, thisutotal, _ = loop_test(itask,
                  p.learn_every, p.stim_on, p.stim_off,
                  p.train_time, p.dt, p.Nsteps, p.Ncells,
                  nothing, nothing, p.LX, p.refrac, learn_step,
                  invtau_bale, invtau_bali, invtau_plas, X_bal, maxTimes,
                  copy_times[Threads.threadid()],
                  copy_ns[Threads.threadid()],
                  copy_timesX[Threads.threadid()],
                  copy_nsX[Threads.threadid()],
                  copy_inputsE[Threads.threadid()],
                  copy_inputsI[Threads.threadid()],
                  copy_inputsP[Threads.threadid()],
                  copy_inputsEPrev[Threads.threadid()],
                  copy_inputsIPrev[Threads.threadid()],
                  copy_inputsPPrev[Threads.threadid()],
                  nothing, nothing, nothing, nothing,
                  copy_u_bale[Threads.threadid()],
                  copy_u_bali[Threads.threadid()],
                  copy_uX_plas[Threads.threadid()],
                  copy_u_bal[Threads.threadid()],
                  copy_u[Threads.threadid()],
                  nothing, nothing,
                  copy_X[Threads.threadid()],
                  p.wid, p.example_neurons,
                  copy_lastSpike[Threads.threadid()],
                  nothing, nothing, nothing, nothing, nothing,
                  copy_v[Threads.threadid()],
                  copy_rng[Threads.threadid()],
                  copy_noise[Threads.threadid()],
                  nothing, sig, nothing, w0Index, w0Weights, nc0, X_stim, nothing,
                  nothing, wpIndexOut, nothing, wpWeightX, nothing, wpWeightOut, nothing,
                  ncpOut, nothing, nothing, nothing, cellModel_args)
            nss[itrial, itask] = thisns[ineurons_to_test]
            timess[itrial, itask] = thistimes[ineurons_to_test,:]
            utotals[itrial, itask] = thisutotal[:,ineurons_to_test]
            println("trial #", itrial, ", task #", itask, ": ",round(t, sigdigits=3), " sec")
        end
    end

    save(joinpath(data_dir,"test.jld2"),
         "ineurons_to_test", ineurons_to_test,
         "nss", nss, "timess", timess, "utotals", utotals)

    no_plot || plot(joinpath(data_dir, "test.jld2"), ineurons_to_plot = ineurons_to_test)

    return (; nss, timess, utotals)
end
