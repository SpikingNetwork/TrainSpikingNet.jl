function test(; ntrials = 1,
                ineurons_to_test = 1:16,
                restore_from_checkpoint = nothing,
                no_plot = false)

    Threads.nthreads() < ndevices() && @warn "performance is best if no. threads is set to no. GPUs"

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
    wpWeightX = load(joinpath(data_dir,"wpWeightX.jld2"), "wpWeightX")
    wpWeightIn = load(joinpath(data_dir,"wpWeightIn-ckpt$R.jld2"))["wpWeightIn"]

    wpWeightOut = zeros(TCharge, maximum([length(x) for x in wpIndexOut])+1, p.Ncells+1);

    w0Index = vv2m(w0Index);
    w0Weights = vv2m(w0Weights);
    wpIndexIn = vv2m(wpIndexIn);
    wpIndexOut = vv2m(wpIndexOut);
    wpIndexConvert = vv2m(wpIndexConvert);
    wpWeightIn = vv2m(wpWeightIn);

    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn);

    # --- set up variables --- #
    global X_stim = CuArray{TCurrent}(X_stim);
    global w0Index = CuArray{p.IntPrecision}(w0Index);
    global w0Weights = CuArray{TCharge}(w0Weights);
    global wpIndexOut = CuArray{p.IntPrecision}(wpIndexOut);
    global wpWeightOut = CuArray{TCharge}(wpWeightOut);
    global wpWeightX = CuArray{TCharge}(wpWeightX);

    rng = eval(p.rng_func.gpu)

    # --- test the network --- #
    ntasks = size(X_stim,3)
    nss = Array{Any}(undef, ntrials, ntasks);
    timess = Array{Any}(undef, ntrials, ntasks);
    utotals = Array{Any}(undef, ntrials, ntasks);

    itrial0 = Threads.Atomic{Int}(1)
    @sync for idevice = 1:ndevices()
        Threads.@spawn begin
            device!(idevice-1)

            copy_rng = typeof(rng)()
            isnothing(p.seed) || Random.seed!.(copy_rng, p.seed + idevice)
            save(joinpath(data_dir,"rng-test-device$idevice.jld2"), "rng", copy_rng)
            copy_X_bal = copy(X_bal)
            copy_sig = copy(sig)
            copy_X_stim = copy(X_stim)
            copy_bnotrefrac = copy(bnotrefrac)
            copy_bspike = copy(bspike)
            copy_bspikeX = copy(bspikeX)
            copy_w0Index = copy(w0Index)
            copy_w0Weights = copy(w0Weights)
            copy_wpWeightX = copy(wpWeightX)
            copy_wpIndexOut = copy(wpIndexOut)
            copy_wpWeightOut = copy(wpWeightOut)
            copy_scratch = typeof(scratch)()
            typeof(p.tau_plas)<:AbstractArray && (copy_invtau_plas = copy(invtau_plas))
            copy_cellModel_args = deepcopy(cellModel_args)

            while true
                itrial = Threads.atomic_add!(itrial0, 1)
                itrial > ntrials && break
                for itask = 1:ntasks
                    t = @elapsed thisns, thistimes, _, _, thisutotal, _ = loop(Val(:test),
                          TCurrent, TCharge, TTime,
                          itask,
                          p.learn_every, p.stim_on, p.stim_off,
                          p.train_time, p.dt, p.Nsteps, p.Ncells,
                          nothing, p.LX, p.refrac, learn_step, learn_nsteps, invtau_bale,
                          invtau_bali,
                          typeof(p.tau_plas)<:Number ? invtau_plas : copy_invtau_plas,
                          copy_X_bal,
                          maxTimes,
                          copy_sig,
                          p.wid, p.example_neurons,
                          plusone,
                          nothing,
                          copy_cellModel_args,
                          copy_bnotrefrac,
                          copy_bspike,
                          copy_bspikeX,
                          copy_scratch,
                          nothing, nothing, nothing, nothing, nothing, nothing,
                          copy_rng,
                          nothing,
                          copy_X_stim,
                          nothing, nothing,
                          copy_w0Index,
                          copy_w0Weights,
                          copy_wpWeightX,
                          nothing,
                          copy_wpIndexOut,
                          nothing,
                          nothing,
                          copy_wpWeightOut);
                    nss[itrial, itask] = Array(thisns[ineurons_to_test])
                    timess[itrial, itask] = Array(thistimes[ineurons_to_test,:])
                    utotals[itrial, itask] = Array(thisutotal[:,ineurons_to_test])
                    println("trial #", itrial, ", task #", itask, ": ",round(t, sigdigits=3), " sec")
                end
            end
        end
    end

    save(joinpath(data_dir, "test.jld2"),
         "ineurons_to_test", ineurons_to_test,
         "nss", nss, "timess", timess, "utotals", utotals,
         "init_code", init_code)

    no_plot || plot(joinpath(data_dir, "test.jld2"), ineurons_to_plot = ineurons_to_test)

    return (; nss, timess, utotals)
end
