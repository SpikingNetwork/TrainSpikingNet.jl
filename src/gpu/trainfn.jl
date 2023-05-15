function train(; nloops = 1,
                 correlation_interval = 1,
                 save_best_checkpoint = false,
                 restore_from_checkpoint = nothing,
                 monitor_resources_used = nothing,
                 return_P = false)

    # --- load initialization --- #
    X_stim = load(joinpath(data_dir,"X_stim.jld2"), "X_stim");
    utarg = load(joinpath(data_dir,"utarg.jld2"), "utarg");
    w0Index = load(joinpath(data_dir,"w0Index.jld2"), "w0Index");
    w0Weights = load(joinpath(data_dir,"w0Weights.jld2"), "w0Weights");
    wpIndexIn = load(joinpath(data_dir,"wpIndexIn.jld2"), "wpIndexIn");
    wpIndexOut = load(joinpath(data_dir,"wpIndexOut.jld2"), "wpIndexOut");
    wpIndexConvert = load(joinpath(data_dir,"wpIndexConvert.jld2"), "wpIndexConvert");
    rateX = load(joinpath(data_dir,"rateX.jld2"), "rateX");
    if isnothing(restore_from_checkpoint)
        R=0
        wpWeightX = load(joinpath(data_dir,"wpWeightX.jld2"), "wpWeightX");
        wpWeightIn = vv2m(load(joinpath(data_dir,"wpWeightIn.jld2"), "wpWeightIn"));
        P = vm2a(load(joinpath(data_dir,"P.jld2"), "P"));
        if p.PPrecision<:Integer
            P .= round.(P .* p.PScale)
        end
    else
        R = typeof(restore_from_checkpoint)<:AbstractString ?
            parse(Int, restore_from_checkpoint[1:end-1]) : restore_from_checkpoint
        wpWeightX = load(joinpath(data_dir,"wpWeightX-ckpt$restore_from_checkpoint.jld2"),
                         "wpWeightX");
        wpWeightIn = load(joinpath(data_dir,"wpWeightIn-ckpt$restore_from_checkpoint.jld2"),
                          "wpWeightIn");
        P = load(joinpath(data_dir,"P-ckpt$restore_from_checkpoint.jld2"), "P");
    end;

    wpWeightOut = zeros(TCharge, maximum([length(x) for x in wpIndexOut])+1, p.Ncells+1);

    w0Index = vv2m(w0Index);
    w0Weights = vv2m(w0Weights);
    wpIndexIn = vv2m(wpIndexIn);
    wpIndexOut = vv2m(wpIndexOut);
    wpIndexConvert = vv2m(wpIndexConvert);

    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn);

    rng = eval(p.rng_func.gpu)
    isnothing(p.seed) || Random.seed!(rng, p.seed)
    save(joinpath(data_dir,"rng-train.jld2"), "rng",rng)

    ntasks = size(utarg,3)

    # --- set up variables --- #
    P = CuArray{p.PPrecision}(P);
    X_stim = CuArray{TCurrent}(X_stim);
    utarg = CuArray{TCurrent}(utarg);
    rateX = CuArray{p.FloatPrecision}(rateX);
    w0Index = CuArray{p.IntPrecision}(w0Index);
    w0Weights = CuArray{TCharge}(w0Weights);
    wpIndexIn = CuArray{p.IntPrecision}(wpIndexIn);
    wpIndexConvert = CuArray{p.IntPrecision}(wpIndexConvert);
    wpIndexOut = CuArray{p.IntPrecision}(wpIndexOut);
    wpWeightIn = CuArray{TCharge}(wpWeightIn);
    wpWeightOut = CuArray{TCharge}(wpWeightOut);
    wpWeightX = CuArray{TCharge}(wpWeightX);

    pLtot = size(wpIndexIn,1) + p.LX
    raug = CuArray{TInvTime}(undef, pLtot, p.Ncells)
    k = CuArray{p.FloatPrecision}(undef, pLtot, p.Ncells)
    vPv = CuArray{TInvTime}(undef, p.Ncells)
    den = CuArray{TTime}(undef, p.Ncells)
    e = CuArray{TCurrent}(undef, p.Ncells)
    delta = CuArray{TCharge}(undef, pLtot, p.Ncells)

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

    function save_weights_P(ckpt)
        save(joinpath(data_dir,"wpWeightX-ckpt$ckpt.jld2"),
             "wpWeightX", Array(wpWeightX))
        save(joinpath(data_dir,"wpWeightIn-ckpt$ckpt.jld2"),
             "wpWeightIn", Array(wpWeightIn))
        save(joinpath(data_dir,"P-ckpt$ckpt.jld2"),
             "P", p.PType==Symmetric ? make_symmetric(Array(P)) : Array(P))
    end

    iloop = 0
    correlation = Union{Missing,Float64}[]
    elapsed_time = Float64[]
    firing_rate = TTime <: Real ? Float64[] : Vector{eltype(unit(p.maxrate)*1.0)}(undef, 0)
    try
        maxcor = -Inf
        for outer iloop = R.+(1:nloops)
            itask = choose_task(iloop, ntasks)
            println("Loop no. ", iloop, ", task no. ", itask) 

            start_time = time()

            if mod(iloop, correlation_interval) != 0

                loop(Val(:train), TCurrent, TCharge, TTime, itask, p.learn_every,
                     p.stim_on, p.stim_off, p.train_time, p.dt, p.Nsteps,
                     p.Ncells, nothing, p.LX, p.refrac, learn_step, learn_nsteps, invtau_bale,
                     invtau_bali, invtau_plas, X_bal, nothing, sig, nothing,
                     nothing, plusone, p.PScale, cellModel_args, bnotrefrac,
                     bspike, bspikeX, scratch, raug, k, vPv, den, e, delta, rng,
                     P, X_stim, utarg, rateX, w0Index, w0Weights, wpWeightX,
                     wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn,
                     wpWeightOut)

                push!(correlation, missing)
            else
                _, _, _, _, utotal, _, _, uplastic, _ = loop(Val(:train_test),
                    TCurrent, TCharge, TTime, itask, p.learn_every, p.stim_on,
                    p.stim_off, p.train_time, p.dt, p.Nsteps, p.Ncells, nothing,
                    p.LX, p.refrac, learn_step, learn_nsteps, invtau_bale, invtau_bali,
                    invtau_plas, X_bal, maxTimes, sig, p.wid, p.example_neurons,
                    plusone, p.PScale, cellModel_args, bnotrefrac, bspike,
                    bspikeX, scratch, raug, k, vPv, den, e, delta, rng, P, X_stim,
                    utarg, rateX, w0Index, w0Weights, wpWeightX, wpIndexIn,
                    wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut)

                if p.correlation_var == :utotal
                    ulearned = utotal
                elseif p.correlation_var == :uplastic
                    ulearned = uplastic
                else
                    error("invalid value for correlation_var parameter")
                end
                pcor = Array{Float64}(undef, p.Ncells)
                for ci in 1:p.Ncells
                    utarg_slice = @view utarg[:,ci, itask]
                    ulearned_slice = @view ulearned[:,ci]
                    if TCurrent <: Real
                        pcor[ci] = cor(convert(Array{Float64}, utarg_slice),
                                       Array(ulearned_slice))
                    else
                        pcor[ci] = cor(convert(Array{Float64}, ustrip(utarg_slice)),
                                       Array(ustrip(ulearned_slice)))
                    end
                end

                bnotnan = .!isnan.(pcor)
                thiscor = mean(pcor[bnotnan])
                println("correlation: ", thiscor,
                        all(bnotnan) ? "" : string(" (", length(pcor)-count(bnotnan)," are NaN)"))
                push!(correlation, thiscor)

                if save_best_checkpoint && thiscor>maxcor && all(bnotnan)
                    suffix = string("ckpt", iloop, "-cor", round(thiscor, digits=3))
                    save_weights_P(suffix)
                    if maxcor != -Inf
                        for oldckptfile in filter(x -> !contains(x, string("ckpt", iloop)) &&
                                                  contains(x, string("-cor", round(maxcor, digits=3))),
                                                  readdir(data_dir))
                            rm(joinpath(data_dir, oldckptfile))
                        end
                    end
                    maxcor = max(maxcor, thiscor)
                end
            end

            iloop == R+nloops && save_weights_P(iloop)

            this_elapsed_time = time()-start_time
            println("elapsed time: ", this_elapsed_time, " sec")
            push!(elapsed_time, this_elapsed_time)

            mean_rate = mean(scratch.ns) / (p.dt*p.Nsteps)
            if TTime <: Real
                mean_rate_conv = string(1000*mean_rate, " Hz")
            else
                mean_rate_conv = uconvert(unit(p.maxrate), mean_rate)
            end
            println("firing rate: ", mean_rate_conv)
            push!(firing_rate, TTime <: Real ? 1000*mean_rate
                                             : uconvert(unit(p.maxrate), mean_rate))
        end
    catch e
        println("stopping early: ", e)
        save_weights_P(string(iloop,'i'))
    finally
        save(joinpath(data_dir,"learning-curve.jld2"),
             "correlation", correlation,
             "elapsed_time", elapsed_time,
             "firing_rate", firing_rate)
    end

    if !isnothing(monitor_resources_used)
      sleep(60)
      close(chnl)
    end

    return (; wpWeightIn, wpWeightX, :P => return_P ? P : nothing)
end
