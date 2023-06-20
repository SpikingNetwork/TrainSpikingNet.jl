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
        ckpts = [(parse(Int, m.captures[1]), m.captures[2])
                 for m in match.(r"ckpt([0-9]+)(.*?)\.jld2",
                                 filter(startswith("wpWeightIn-ckpt"),
                                        readdir(data_dir)))]
        R = join(argmax(first, ckpts))
        println("using checkpoint ", R)
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

    chan_rng = Channel{typeof(rng)}(Threads.nthreads())
    copy_rng = [typeof(rng)() for _=1:Threads.nthreads()];
    for x in copy_rng;  put!(chan_rng, x);  end
    isnothing(p.seed) || Random.seed!.(copy_rng, p.seed .+ (1:Threads.nthreads()))
    save(joinpath(data_dir,"rng-test.jld2"), "rng", copy_rng)

    chan_scratch = Channel{typeof(scratch)}(Threads.nthreads())
    for _=1:Threads.nthreads(); put!(chan_scratch, typeof(scratch)()); end

    itrial0 = Threads.Atomic{Int}(1)
    try
        @sync for ithread = 1:Threads.nthreads()
            Threads.@spawn begin
                this_rng = take!(chan_rng)
                this_scratch = take!(chan_scratch)
                while true
                    itrial = Threads.atomic_add!(itrial0, 1)
                    itrial > ntrials && break
                    for itask = 1:ntasks
                        t = @elapsed thisns, thistimes, _, _, thisutotal, _ = loop(Val(:test),
                              TCurrent, TCharge, TTime, itask, p.learn_every, p.stim_on,
                              p.stim_off, p.train_time, p.dt, p.Nsteps, nothing, nothing,
                              p.Ncells, nothing, p.LX, p.refrac, learn_step, learn_nsteps, invtau_bale,
                              invtau_bali, invtau_plas, X_bal, maxTimes, sig, p.wid,
                              p.example_neurons, nothing, nothing, nothing, cellModel_args,
                              nothing, nothing, this_scratch, nothing,
                              nothing, nothing, this_rng, nothing,
                              X_stim, nothing, nothing, w0Index, w0Weights, nothing,
                              wpIndexOut, nothing, wpWeightX, nothing, wpWeightOut)

                        nss[itrial, itask] = copy(thisns[ineurons_to_test])
                        timess[itrial, itask] = copy(thistimes[ineurons_to_test,:])
                        utotals[itrial, itask] = copy(thisutotal[:,ineurons_to_test])
                        println("trial #", itrial, ", task #", itask, ": ",round(t, sigdigits=3), " sec")
                    end
                end
                put!(chan_rng, this_rng)
                put!(chan_scratch, this_scratch)
            end
        end
    catch e
        println("stopping early: ", e)
    finally
        # discard unfinished trials
        inotassigned = [!isassigned(nss, i) for i in eachindex(nss)]
        nss[inotassigned] = timess[inotassigned] = utotals[inotassigned] .= missing
        itrial_keep = .![all(ismissing.(x)) for x in eachrow(nss)]
        nss = nss[itrial_keep,:]
        timess = timess[itrial_keep,:]
        utotals = utotals[itrial_keep,:]

        # convert ragged spike times matrix to vector of vectors
        global timess_vec = similar(timess)
        _eltype = eltype(timess[1])
        nneurons = length(nss[1])
        for itrialtask in eachindex(nss)
            timess_vec[itrialtask] = Vector{Vector{_eltype}}(undef, nneurons)
            for ineuron = eachindex(nss[itrialtask])
                nspikes = nss[itrialtask][ineuron]
                timess_vec[itrialtask][ineuron] = timess[itrialtask][ineuron, 1:nspikes]
            end
        end

        save(joinpath(data_dir,"test.jld2"),
             "ineurons_to_test", ineurons_to_test,
             "times", timess_vec, "utotal", utotals,
             "init_code", init_code)

        Threads.atomic_add!(itrial0, ntrials+1)
    end

    no_plot || plot(joinpath(data_dir, "test.jld2"), ineurons_to_plot = ineurons_to_test)

    return (; times=timess_vec, utotal=utotals)
end
