function train(; nloops = 1,
                 correlation_interval = 1,
                 save_best_checkpoint = false,
                 restore_from_checkpoint = nothing,
                 monitor_resources_used = nothing,
                 return_P = false)

    # --- load initialization --- #
    w0Index = load(joinpath(data_dir,"w0Index.jld2"), "w0Index");
    w0Weights = load(joinpath(data_dir,"w0Weights.jld2"), "w0Weights");
    nc0 = load(joinpath(data_dir,"nc0.jld2"), "nc0");
    ncpIn = load(joinpath(data_dir,"ncpIn.jld2"), "ncpIn");
    ncpOut = load(joinpath(data_dir,"ncpOut.jld2"), "ncpOut");
    X_stim = load(joinpath(data_dir,"X_stim.jld2"), "X_stim");
    utarg = load(joinpath(data_dir,"utarg.jld2"), "utarg");
    wpIndexIn = load(joinpath(data_dir,"wpIndexIn.jld2"), "wpIndexIn");
    wpIndexOut = load(joinpath(data_dir,"wpIndexOut.jld2"), "wpIndexOut");
    wpIndexConvert = load(joinpath(data_dir,"wpIndexConvert.jld2"), "wpIndexConvert");
    rateX = load(joinpath(data_dir,"rateX.jld2"), "rateX");
    if isnothing(restore_from_checkpoint);
        R=0
        wpWeightX = load(joinpath(data_dir,"wpWeightX.jld2"), "wpWeightX");
        wpWeightIn = load(joinpath(data_dir,"wpWeightIn.jld2"), "wpWeightIn");
        Pinv_norm = load(joinpath(data_dir,"P.jld2"), "P");
        P = Vector{p.PType}();
        for ci=1:p.Ncells
            if p.PPrecision<:AbstractFloat
                push!(P, copy(Pinv_norm));
            else
                push!(P, round.(Pinv_norm * p.PScale));
            end
        end
    else
        R = restore_from_checkpoint;
        wpWeightX = load(joinpath(data_dir,"wpWeightX-ckpt$R.jld2"), "wpWeightX");
        wpWeightIn = load(joinpath(data_dir,"wpWeightIn-ckpt$R.jld2"), "wpWeightIn");
        P = load(joinpath(data_dir,"P-ckpt$R.jld2"), "P");
    end;
    wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells);
    wpWeightOut = convertWgtIn2Out(p.Ncells, ncpIn,
                                   wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut);

    rng = eval(p.rng_func.cpu)
    isnothing(p.seed) || Random.seed!(rng, p.seed)
    save(joinpath(data_dir,"rng-train.jld2"), "rng", rng)

    ntasks = size(utarg,3)

    # --- set up variables --- #
    PType = typeof(p.PType(p.PPrecision.([1. 2; 3 4])));
    P = Vector{PType}(P);
    X_stim = Array{p.FloatPrecision}(X_stim);
    utarg = Array{p.FloatPrecision}(utarg);
    nc0 = Array{p.IntPrecision}(nc0);
    ncpIn = Array{p.IntPrecision}(ncpIn);
    ncpOut = Array{p.IntPrecision}(ncpOut);
    w0Index = Array{p.IntPrecision}(w0Index);
    w0Weights = Array{p.FloatPrecision}(w0Weights);
    wpIndexIn = Array{p.IntPrecision}(wpIndexIn);
    wpIndexConvert = Array{p.IntPrecision}(wpIndexConvert);
    wpIndexOut = Array{p.IntPrecision}(wpIndexOut);
    wpWeightIn = Array{p.FloatPrecision}(wpWeightIn);
    wpWeightX = Array{p.FloatPrecision}(wpWeightX);
    wpWeightOut = Array{p.FloatPrecision}(wpWeightOut);
    rateX = Array{p.FloatPrecision}(rateX);

    # --- monitor resources used --- #
    function monitor_resources(c::Channel)
      while true
        isopen(c) || break
        ipmitool = readlines(pipeline(`sudo ipmitool sensor`,
                                      `grep "PW Consumption"`,
                                      `cut -d'|' -f2`))
        top = readlines(pipeline(`top -b -n 2 -p $(getpid())`,
                                 `tail -1`,
                                 `awk '{print $9; print $10}'`))
        println("total power used: ", strip(ipmitool[1]), " Watts\n",
                "CPU cores used by this process: ", strip(top[1]), "%\n",
                "CPU memory used by this process: ", strip(top[2]), '%')
        sleep(monitor_resources_used)
      end
    end

    if !isnothing(monitor_resources_used)
      chnl = Channel(monitor_resources)
      sleep(60)
    end

    # --- train the network --- #
    maxcor = -Inf
    for iloop = R.+(1:nloops)
        itask = choose_task(iloop, ntasks)
        println("Loop no. ", iloop, ", task no. ", itask) 

        start_time = time()

        if mod(iloop, correlation_interval) != 0

            loop_train(itask,
                p.learn_every, p.stim_on, p.stim_off,
                p.train_time, p.dt, p.Nsteps, p.Ncells,
                nothing, p.Lexc+p.Linh, p.LX, p.refrac,
                learn_step, invtau_bale, invtau_bali, invtau_plas, X_bal,
                nothing, nothing, ns, nothing, nsX, inputsE,
                inputsI, inputsP, inputsEPrev, inputsIPrev, inputsPPrev,
                spikes, spikesPrev, spikesX, spikesXPrev, u_bale, u_bali,
                uX_plas, u_bal, u, r, rX, X, nothing, nothing,
                lastSpike, plusone, exactlyzero, p.PScale, raug, k, v, rng, noise,
                rndX, sig, P, w0Index, w0Weights, nc0, X_stim, utarg,
                wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightX, wpWeightIn,
                wpWeightOut, ncpIn, ncpOut, nothing, nothing, rateX,
                cellModel_args)
        else
            _, _, _, _, utotal, _, _, uplastic, _ = loop_train_test(itask,
                p.learn_every, p.stim_on, p.stim_off,
                p.train_time, p.dt, p.Nsteps, p.Ncells,
                nothing, p.Lexc+p.Linh, p.LX, p.refrac,
                learn_step, invtau_bale, invtau_bali, invtau_plas, X_bal,
                maxTimes, times, ns, timesX, nsX, inputsE, inputsI,
                inputsP, inputsEPrev, inputsIPrev, inputsPPrev, spikes,
                spikesPrev, spikesX, spikesXPrev, u_bale, u_bali,
                uX_plas, u_bal, u, r, rX, X, p.wid,
                p.example_neurons, lastSpike, plusone, exactlyzero, p.PScale,
                raug, k, v, rng, noise, rndX, sig, P, w0Index, w0Weights,
                nc0, X_stim, utarg, wpIndexIn, wpIndexOut, wpIndexConvert,
                wpWeightX, wpWeightIn, wpWeightOut, ncpIn, ncpOut, nothing,
                nothing, rateX, cellModel_args)

            if p.correlation_var == :utotal
                ulearned = utotal
            elseif p.correlation_var == :uplastic
                ulearned = uplastic
            else
                error("invalid value for correlation_var peter")
            end
            pcor = Array{Float64}(undef, p.Ncells)
            for ci in 1:p.Ncells
                utarg_slice = @view utarg[:,ci, itask]
                ulearned_slice = @view ulearned[:,ci]
                pcor[ci] = cor(utarg_slice, ulearned_slice)
            end

            bnotnan = .!isnan.(pcor)
            thiscor = mean(pcor[bnotnan])
            println("correlation: ", thiscor,
                    all(bnotnan) ? "" : string(" (", length(pcor)-count(bnotnan)," are NaN)"))

            if save_best_checkpoint && thiscor>maxcor && all(bnotnan)
                suffix = string("ckpt", iloop, "-cor", round(thiscor, digits=3))
                save(joinpath(data_dir, "wpWeightIn-$suffix.jld2"),
                     "wpWeightIn", Array(wpWeightIn))
                save(joinpath(data_dir, "wpWeightX-$suffix.jld2"),
                     "wpWeightX", Array(wpWeightX))
                save(joinpath(data_dir, "P-$suffix.jld2"), "P", P)
                if maxcor != -Inf
                    for oldckptfile in filter(x -> !contains(x, string("ckpt", iloop)) &&
                                              contains(x, string("-cor", round(maxcor, digits=3))),
                                              readdir(data_dir))
                        rm(joinpath(data_dir, oldckptfile))
                    end
                end
                global maxcor = max(maxcor, thiscor)
            end
        end

        if iloop == R+nloops
            save(joinpath(data_dir,"wpWeightIn-ckpt$iloop.jld2"),
                 "wpWeightIn", Array(wpWeightIn))
            save(joinpath(data_dir,"wpWeightX-ckpt$iloop.jld2"),
                 "wpWeightX", Array(wpWeightX))
            save(joinpath(data_dir,"P-ckpt$iloop.jld2"), "P", P)
        end


        elapsed_time = time()-start_time
        println("elapsed time: ", elapsed_time, " sec")
        println("firing rate: ", mean(ns) / (p.dt/1000*p.Nsteps), " Hz")
    end

    if !isnothing(monitor_resources_used)
      sleep(60)
      close(chnl)
    end

    return (; wpWeightIn, wpWeightX, :P => return_P ? P : nothing)
end
