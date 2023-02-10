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
    X_stim = load(joinpath(data_dir,"X_stim.jld2"), "X_stim");
    utarg = load(joinpath(data_dir,"utarg.jld2"), "utarg");
    wpIndexIn = load(joinpath(data_dir,"wpIndexIn.jld2"), "wpIndexIn");
    wpIndexOut = load(joinpath(data_dir,"wpIndexOut.jld2"), "wpIndexOut");
    wpIndexConvert = load(joinpath(data_dir,"wpIndexConvert.jld2"), "wpIndexConvert");
    rateX = load(joinpath(data_dir,"rateX.jld2"), "rateX");
    if isnothing(restore_from_checkpoint)
        R=0
        wpWeightX = load(joinpath(data_dir,"wpWeightX.jld2"), "wpWeightX");
        wpWeightIn = load(joinpath(data_dir,"wpWeightIn.jld2"), "wpWeightIn");
        Pinv_norm = load(joinpath(data_dir,"P.jld2"), "P");
        PLtot = p.Lexc + p.Linh + p.LX
        P1 = p.PType(Array{Float64}(undef, PLtot, PLtot));
        P = Array{Float64}(undef, (size(p.PType==SymmetricPacked ? P1.tri : P1)..., p.Ncells));
        P .= p.PType==SymmetricPacked ? Pinv_norm.tri : Pinv_norm;
        if p.PPrecision<:Integer
            P .= round.(P .* p.PScale)
        end
    else
        R = restore_from_checkpoint
        wpWeightX = load(joinpath(data_dir,"wpWeightX-ckpt$R.jld2"), "wpWeightX");
        wpWeightIn = load(joinpath(data_dir,"wpWeightIn-ckpt$R.jld2"), "wpWeightIn");
        P = load(joinpath(data_dir,"P-ckpt$R.jld2"), "P");
    end;
    wpWeightOut = zeros(maximum(wpIndexConvert), p.Ncells);
    wpWeightOut = convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut);

    rng = eval(p.rng_func.gpu)
    isnothing(p.seed) || Random.seed!(rng, p.seed)
    save(joinpath(data_dir,"rng-train.jld2"), "rng",rng)

    ntasks = size(utarg,3)

    # --- set up variables --- #
    P = CuArray{p.PPrecision}(P);
    nc0 = CuArray{p.IntPrecision}(nc0);
    X_stim = CuArray{p.FloatPrecision}(X_stim);
    utarg = CuArray{p.FloatPrecision}(utarg);
    w0Index = CuArray{p.IntPrecision}(w0Index);
    w0Weights = CuArray{p.FloatPrecision}(w0Weights);
    wpIndexIn = CuArray{p.IntPrecision}(wpIndexIn);
    wpIndexConvert = CuArray{p.IntPrecision}(wpIndexConvert);
    wpIndexOut = CuArray{p.IntPrecision}(wpIndexOut);
    wpWeightX = CuArray{p.FloatPrecision}(wpWeightX);
    wpWeightIn = CuArray{p.FloatPrecision}(wpWeightIn);
    wpWeightOut = CuArray{p.FloatPrecision}(wpWeightOut);
    rateX = CuArray{p.FloatPrecision}(rateX);

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
        nvidiasmi = readlines(pipeline(`nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory --format=csv,noheader,nounits`))
        data = dropdims(sum(hcat([parse.(Float64, split(strip(x), ',')) for x in nvidiasmi]...),
                            dims=2), dims=2)
        println("total power used: ", strip(ipmitool[1]), " Watts\n",
                "CPU cores used by this process: ", strip(top[1]), "%\n",
                "CPU memory used by this process: ", strip(top[2]), "%\n",
                "GPU power used: ", data[1], " Watts\n",
                "GPU cores used: ", data[2], "%\n",
                "GPU memory used: ", data[3], "%")
        sleep(monitor_resources_used)
      end
    end

    if !isnothing(monitor_resources_used)
      chnl = Channel(monitor_resources)
      sleep(60)
    end

    # --- train the network --- #
    function make_symmetric(A::Array)
        for i=1:size(A,1), j=i+1:size(A,2)
            A[j,i,:] = A[i,j,:]
        end
        return A
    end

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
                lastSpike, bnotrefrac, bspike, plusone, minusone, p.PScale, raug,
                k, den, e, delta, v, rng, noise, rndX, sig, P, w0Index,
                w0Weights, nc0, X_stim, utarg, wpWeightX, wpIndexIn, wpIndexOut,
                wpIndexConvert, wpWeightIn, wpWeightOut, rateX,
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
                p.example_neurons, lastSpike, bnotrefrac, bspike, plusone,
                minusone, p.PScale, raug, k, den, e, delta, v, rng, noise, rndX,
                sig, P, w0Index, w0Weights, nc0, X_stim, utarg, wpWeightX,
                wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut,
                rateX, cellModel_args)

            if p.correlation_var == :utotal
                ulearned = utotal
            elseif p.correlation_var == :uplastic
                ulearned = uplastic
            else
                error("invalid value for correlation_var parameter")
            end
            pcor = Array{Float64}(undef, p.Ncells)
            for ci in 1:p.Ncells
                utarg_slice = convert(Array{Float64}, utarg[:,ci, itask])
                ulearned_slice = Array(ulearned[:,ci])
                pcor[ci] = cor(utarg_slice, ulearned_slice)
            end

            bnotnan = .!isnan.(pcor)
            thiscor = mean(pcor[bnotnan])
            println("correlation: ", thiscor,
                    all(bnotnan) ? "" : string(" (", length(pcor)-count(bnotnan)," are NaN)"))

            if save_best_checkpoint && thiscor>maxcor && all(bnotnan)
                suffix = string("ckpt", iloop, "-cor", round(thiscor, digits=3))
                save(joinpath(data_dir, "wpWeightX-$suffix.jld2"),
                     "wpWeightX", Array(wpWeightX))
                save(joinpath(data_dir, "wpWeightIn-$suffix.jld2"),
                     "wpWeightIn", Array(wpWeightIn))
                save(joinpath(data_dir, "P-$suffix.jld2"),
                     "P", p.PType==Symmetric ? make_symmetric(Array(P)) : Array(P))
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
            save(joinpath(data_dir,"wpWeightX-ckpt$iloop.jld2"),
                 "wpWeightX", Array(wpWeightX))
            save(joinpath(data_dir,"wpWeightIn-ckpt$iloop.jld2"),
                 "wpWeightIn", Array(wpWeightIn))
            save(joinpath(data_dir,"P-ckpt$iloop.jld2"),
                 "P", p.PType==Symmetric ? make_symmetric(Array(P)) : Array(P))
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
