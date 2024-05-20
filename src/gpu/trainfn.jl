function train(; nloops = 1,
                 correlation_interval = 1,
                 save_best_checkpoint = false,
                 restore_from_checkpoint = nothing,
                 monitor_resources_used = nothing,
                 return_P_rrXhistory  = false)

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
        if p.PCompute == :fast
            P = vm2a(load(joinpath(data_dir,"P.jld2"), "P"));
            if p.PPrecision<:Integer
                P .= round.(P .* p.PScale)
            end
        else
            P = nothing
            empty!(scratch.rrXhistory)
        end
    else
        R = typeof(restore_from_checkpoint)<:AbstractString ?
            parse(Int, restore_from_checkpoint[1:end-1]) : restore_from_checkpoint
        wpWeightX = load(joinpath(data_dir,"wpWeightX-ckpt$restore_from_checkpoint.jld2"),
                         "wpWeightX");
        wpWeightIn = load(joinpath(data_dir,"wpWeightIn-ckpt$restore_from_checkpoint.jld2"),
                          "wpWeightIn");
        if p.PCompute == :fast
            P = load(joinpath(data_dir,"P-ckpt$restore_from_checkpoint.jld2"), "P");
        else
            P = nothing
            empty!(scratch.rrXhistory)
            for rrXi in eachcol(load(joinpath(data_dir,
                                              "rrXhistory-ckpt$restore_from_checkpoint.jld2"),
                                     "rrXhistory"))
                push!(scratch.rrXhistory, CuArray(rrXi))
            end
        end
    end;

    wpWeightOut = zeros(TCharge, maximum([length(x) for x in wpIndexOut])+1, p.Ncells+1);

    w0Index = vv2m(w0Index);                w0Index .+= 0x1
    w0Weights = vv2m(w0Weights);
    wpIndexIn = vv2m(wpIndexIn);            wpIndexIn .+= 0x1
    wpIndexOut = vv2m(wpIndexOut);          wpIndexOut .+= 0x1
    wpIndexConvert = vv2m(wpIndexConvert);  wpIndexConvert .+= 0x1

    rng = eval(p.rng_func.gpu)
    isnothing(p.seed) || Random.seed!(rng, p.seed)
    save(joinpath(data_dir,"rng-train.jld2"), "rng",rng)

    ntasks = size(utarg,3)

    # --- set up variables --- #
    if p.PCompute == :fast
        P = CuArray{p.PPrecision}(P);
    end
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

    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn);

    # --- dynamically sized scratch space --- #
    pLtot = size(wpIndexIn,1) + p.LX
    raug = CuArray{TInvTime}(undef, pLtot, p.Ncells)
    k = CuArray{p.FloatPrecision}(undef, pLtot, p.Ncells)
    k2 = reshape(k, pLtot, 1, p.Ncells)
    rrXg = CuMatrix{p.PPrecision}(undef, pLtot, p.PComputeN)
    vPv = CuArray{TInvTime}(undef, p.Ncells)
    den = CuArray{TTime}(undef, p.Ncells)
    e = CuArray{TCurrent}(undef, p.Ncells)
    delta = CuArray{TCharge}(undef, pLtot, p.Ncells)
    @static if p.PCompute == :small
        Pinv = CuArray{p.PPrecision}(undef, pLtot, pLtot, p.PComputeN)
        pivot = CuArray{Int32}(undef, pLtot, p.Ncells)
        info = CuArray{Int32}(undef, p.Ncells)
    else
        Pinv = info = pivot = nothing
    end

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

    function save_weights_P_rrXhistory(ckpt)
        save(joinpath(data_dir,"wpWeightX-ckpt$ckpt.jld2"),
             "wpWeightX", Array(wpWeightX))
        save(joinpath(data_dir,"wpWeightIn-ckpt$ckpt.jld2"),
             "wpWeightIn", Array(wpWeightIn))
        if p.PCompute == :fast
            save(joinpath(data_dir,"P-ckpt$ckpt.jld2"),
                 "P", p.PType==Symmetric ? make_symmetric(Array(P)) : Array(P))
        else
            save(joinpath(data_dir,"rrXhistory-ckpt$ckpt.jld2"),
                 "rrXhistory", Array(scratch.rrXhistory.buffer))
        end
    end

    iloop = 0
    correlation = Union{Missing,Float64}[]
    elapsed_time = Float64[]
    firing_rate = TTime <: Real ? Float64[] : Vector{eltype(unit(p.maxrate)*1.0)}(undef, 0)
    try
        maxcor = -Inf
        if TTime <: Real
            rate_unit = "Hz"
        else
            rate_unit = string(unit(p.maxrate))
            fmt = Printf.Format(string("%#", 14+length(rate_unit), "g  "))
        end
        println("loop #  task #  elapsed time (s)  firing rate ($rate_unit)  correlation")
        for outer iloop = R.+(1:nloops)
            itask = choose_task(iloop, ntasks)
            @printf "%6i  %6i  " iloop itask

            start_time = time()

            if mod(iloop, correlation_interval) != 0

                loop(Val(:train), TCurrent, TCharge, TTime, itask, p.learn_every,
                     p.stim_on, p.stim_off, p.train_time, p.dt, p.Nsteps,
                     p.Ncells, nothing, p.LX, p.refrac, learn_step, learn_nsteps, invtau_bale,
                     invtau_bali, invtau_plas, X_bal, nothing, sig, nothing,
                     nothing, plusone, p.PScale, cellModel_args, bnotrefrac,
                     bspike, bspikeX, scratch, raug, k, k2, rrXg, vPv, den, e, delta, rng,
                     P, Pinv, pivot, info, X_stim, utarg, rateX, w0Index, w0Weights, wpWeightX,
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
                    bspikeX, scratch, raug, k, k2, rrXg, vPv, den, e, delta, rng, P, Pinv, pivot, info, X_stim,
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
                num_nan = length(pcor) - count(bnotnan)
                push!(correlation, thiscor)

                if save_best_checkpoint && thiscor>maxcor && all(bnotnan)
                    suffix = string(iloop, "-cor", round(thiscor, digits=3))
                    save_weights_P_rrXhistory(suffix)
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

            iloop == R+nloops && save_weights_P_rrXhistory(iloop)

            this_elapsed_time = time()-start_time
            @printf "%#16g  " this_elapsed_time
            push!(elapsed_time, this_elapsed_time)

            mean_rate = mean(scratch.ns) / (p.dt*p.Nsteps)
            if TTime <: Real
                @printf "%#16g  " 1000*mean_rate
            else
                Printf.format(stdout, fmt, ustrip(uconvert(unit(p.maxrate), mean_rate)))
            end
            push!(firing_rate, TTime <: Real ? 1000*mean_rate
                                             : uconvert(unit(p.maxrate), mean_rate))

            if mod(iloop, correlation_interval) == 0
                @printf "%#11g" thiscor
                num_nan>0 && @printf "%s" string(" (", num_nan, " are NaN)")
            end
            println()
            flush(stdout);  flush(stderr)
        end
    catch e
        println("stopping early")
        showerror(stdout, e, stacktrace(catch_backtrace()))
        save_weights_P_rrXhistory(string(iloop,'i'))
    finally
        save(joinpath(data_dir,"learning-curve.jld2"),
             "correlation", correlation,
             "elapsed_time", elapsed_time,
             "firing_rate", firing_rate,
             "max_memory", Sys.maxrss())
    end

    if !isnothing(monitor_resources_used)
      sleep(60)
      close(chnl)
    end

    return (; wpWeightIn, wpWeightX,
               :P => return_P_rrXhistory && p.PCompute == :fast ? P : nothing,
               :rrXhistory => return_P_rrXhistory && p.PCompute == :small ? scratch.rrXhistory : nothing)
end
