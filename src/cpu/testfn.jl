function test(; ntrials = 1,
                ineurons_to_test = 1:16,
                restore_from_checkpoint = nothing,
                no_plot = false)

    # --- load initialization --- #
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

    wpWeightOut = [Vector{TCharge}(undef, length(x)) for x in wpIndexOut];
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn);

    # --- set up variables --- #
    X_stim = Array{TCurrent}(X_stim);
    w0Index = Vector{Vector{p.IntPrecision}}(w0Index);
    w0Weights = Vector{Vector{TCharge}}(w0Weights);
    wpIndexOut = Vector{Vector{p.IntPrecision}}(wpIndexOut);
    wpWeightOut = Vector{Vector{TCharge}}(wpWeightOut);
    wpWeightX = Array{TCharge}(wpWeightX);

    rng = eval(p.rng_func.cpu)

    # --- test the network --- #
    ntasks = size(X_stim,3)
    nss = Array{Any}(undef, ntrials, ntasks);
    timess = Array{Any}(undef, ntrials, ntasks);
    utotals = Array{Any}(undef, ntrials, ntasks);
    copy_rng = [typeof(rng)() for _=1:Threads.nthreads()];
    isnothing(p.seed) || Random.seed!.(copy_rng, p.seed .+ (1:Threads.nthreads()))
    save(joinpath(data_dir,"rng-test.jld2"), "rng", copy_rng)
    copy_scratch = [typeof(scratch)() for _=1:Threads.nthreads()];
    Threads.@threads :static for itrial=1:ntrials
        for itask = 1:ntasks
            t = @elapsed thisns, thistimes, _, _, thisutotal, _ = loop(itask,
                  p.learn_every, p.stim_on, p.stim_off, p.train_time,
                  p.dt, p.Nsteps, nothing, nothing, p.Ncells, nothing,
                  p.LX, p.refrac, learn_step, invtau_bale, invtau_bali,
                  invtau_plas, X_bal, maxTimes, sig, p.wid, p.example_neurons,
                  nothing, nothing, nothing, cellModel_args, nothing, nothing,
                  copy_scratch[Threads.threadid()], nothing, nothing, nothing,
                  copy_rng[Threads.threadid()], nothing, X_stim, nothing,
                  nothing, w0Index, w0Weights, nothing, wpIndexOut, nothing,
                  wpWeightX, nothing, wpWeightOut, TCurrent, TCharge, TTime)

            nss[itrial, itask] = thisns[ineurons_to_test]
            timess[itrial, itask] = thistimes[ineurons_to_test,:]
            utotals[itrial, itask] = thisutotal[:,ineurons_to_test]
            println("trial #", itrial, ", task #", itask, ": ",round(t, sigdigits=3), " sec")
        end
    end

    save(joinpath(data_dir,"test.jld2"),
         "ineurons_to_test", ineurons_to_test,
         "nss", nss, "timess", timess, "utotals", utotals,
         "init_code", init_code)

    no_plot || plot(joinpath(data_dir, "test.jld2"), ineurons_to_plot = ineurons_to_test)

    return (; nss, timess, utotals)
end
