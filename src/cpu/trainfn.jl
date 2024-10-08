function train(; nloops = 1,
                 correlation_interval = 1,
                 save_best_checkpoint = false,
                 restore_from_checkpoint = nothing,
                 monitor_resources_used = nothing,
                 return_P_raughist = false)

    # --- load initialization --- #
    w0Index = load(joinpath(data_dir,"w0Index.jld2"), "w0Index");
    w0Weights = load(joinpath(data_dir,"w0Weights.jld2"), "w0Weights");
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
        if p.PCompute == :fast
            P = load(joinpath(data_dir,"P.jld2"), "P");
            if p.PPrecision<:Integer
                P = round.(P * p.PScale);
            end
        else
            P = nothing
            empty!(scratch.raughist)
        end
    else
        R = typeof(restore_from_checkpoint)<:AbstractString ?
            parse(Int, restore_from_checkpoint[1:end-1]) : restore_from_checkpoint;
        wpWeightX = load(joinpath(data_dir,"wpWeightX-ckpt$restore_from_checkpoint.jld2"),
                         "wpWeightX");
        wpWeightIn = load(joinpath(data_dir,"wpWeightIn-ckpt$restore_from_checkpoint.jld2"),
                          "wpWeightIn");
        if p.PCompute == :fast
            P = load(joinpath(data_dir,"P-ckpt$restore_from_checkpoint.jld2"), "P");
        else
            P = nothing
            empty!(scratch.raughist)
            for rrXi in eachcol(load(joinpath(data_dir,
                                              "raughist-ckpt$restore_from_checkpoint.jld2"),
                                     "raughist"))
                push!(scratch.raughist, rrXi)
            end
        end
    end;

    wpWeightOut = [Vector{TCharge}(undef, length(x)) for x in wpIndexOut];
    wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn);

    rng = eval(p.rng_func.cpu)
    isnothing(p.seed) || Random.seed!(rng, p.seed)
    save(joinpath(data_dir,"rng-train.jld2"), "rng", rng)

    ntasks = size(utarg,3)

    # --- set up variables --- #
    if p.PCompute == :fast
        PType = typeof(p.PType(p.PPrecision.([1. 2; 3 4])));
        P = Vector{PType}(P);
    end
    X_stim = Array{TCurrent}(X_stim);
    utarg = Array{TCurrent}(utarg);
    rateX = Array{p.FloatPrecision}(rateX);
    w0Index = Vector{Vector{p.IntPrecision}}(w0Index);
    w0Weights = Vector{Vector{TCharge}}(w0Weights);
    wpIndexIn = Vector{Vector{p.IntPrecision}}(wpIndexIn);
    wpIndexConvert = Vector{Vector{p.IntPrecision}}(wpIndexConvert);
    wpIndexOut = Vector{Vector{p.IntPrecision}}(wpIndexOut);
    wpWeightIn = Vector{Vector{TCharge}}(wpWeightIn);
    wpWeightOut = Vector{Vector{TCharge}}(wpWeightOut);
    wpWeightX = Array{TCharge}(wpWeightX);

    # --- dynamically sized scratch space --- #
    pLtot = maximum([length(x) for x in wpIndexIn]) + p.LX
    raug = Matrix{TInvTime}(undef, pLtot, Threads.nthreads())
    k = Matrix{p.FloatPrecision}(undef, pLtot, Threads.nthreads())
    rgather = Matrix{p.PPrecision}(undef, pLtot, Threads.nthreads())
    delta = Matrix{TCharge}(undef, pLtot, Threads.nthreads())
    @static if p.PCompute == :small
        Pinv = Array{p.PPrecision}(undef, pLtot, pLtot, Threads.nthreads())
        @static if p.PType == Array
            work = lwork = nothing
            pivot = [Array{Int64}(undef, pLtot) for _ in 1: Threads.nthreads()]
        else
            _work, lwork, _pivot = get_workspace(BunchKaufman, Symmetric(Pinv[:,:,1]))
            work = [copy(_work) for _ in 1:Threads.nthreads()]
            pivot = [copy(_pivot) for _ in 1:Threads.nthreads()]
        end
    else
        Pinv = work = lwork = pivot = nothing
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
    function save_weights_P_raughist(ckpt)
        save(joinpath(data_dir,"wpWeightIn-ckpt$ckpt.jld2"), "wpWeightIn", Array(wpWeightIn))
        save(joinpath(data_dir,"wpWeightX-ckpt$ckpt.jld2"), "wpWeightX", Array(wpWeightX))
        if p.PCompute == :fast
            save(joinpath(data_dir,"P-ckpt$ckpt.jld2"), "P", P)
        else
            save(joinpath(data_dir,"raughist-ckpt$ckpt.jld2"), "raughist", scratch.raughist)
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
                     nothing, nothing, p.Ncells, nothing, p.LX, p.refrac,
                     learn_step, learn_nsteps, invtau_bale, invtau_bali, invtau_plas, X_bal,
                     nothing, sig, nothing, nothing, plusone, exactlyzero,
                     p.PScale, cellModel_args, uavg, ustd, scratch, raug, k, rgather,
                     delta, rng, P, Pinv, work, lwork, pivot, X_stim, utarg, rateX, w0Index,
                     w0Weights, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightX, wpWeightIn,
                     wpWeightOut)

                push!(correlation, missing)
            else
                _, _, _, _, utotal, _, _, uplastic, _ = loop(Val(:train_test),
                    TCurrent, TCharge, TTime, itask, p.learn_every, p.stim_on,
                    p.stim_off, p.train_time, p.dt, p.Nsteps, nothing,
                    nothing, p.Ncells, nothing, p.LX, p.refrac, learn_step, learn_nsteps,
                    invtau_bale, invtau_bali, invtau_plas, X_bal, maxTimes,
                    sig, p.wid, p.example_neurons, plusone, exactlyzero,
                    p.PScale, cellModel_args, uavg, ustd, scratch, raug, k, rgather,
                    delta, rng, P, Pinv, work, lwork, pivot, X_stim, utarg, rateX, w0Index,
                    w0Weights, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightX, wpWeightIn,
                    wpWeightOut)

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
                        pcor[ci] = cor(utarg_slice, ulearned_slice)
                    else
                        pcor[ci] = cor(ustrip(utarg_slice), ustrip(ulearned_slice))
                    end
                end

                bnotnan = .!isnan.(pcor)
                thiscor = mean(pcor[bnotnan])
                num_nan = length(pcor) - count(bnotnan)
                push!(correlation, thiscor)

                if save_best_checkpoint && thiscor>maxcor && all(bnotnan)
                    suffix = string(iloop, "-cor", round(thiscor, digits=3))
                    save_weights_P_raughist(suffix)
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

            iloop == R+nloops && save_weights_P_raughist(iloop)

            this_elapsed_time = time()-start_time
            @printf "%16.11g  " this_elapsed_time
            push!(elapsed_time, this_elapsed_time)

            mean_rate = mean(scratch.ns) / (p.dt*p.Nsteps)
            if TTime <: Real
                @printf "%16.11g  " 1000*mean_rate
            else
                Printf.format(stdout, fmt, ustrip(uconvert(unit(p.maxrate), mean_rate)))
            end
            push!(firing_rate, TTime <: Real ? 1000*mean_rate
                                             : uconvert(unit(p.maxrate), mean_rate))

            if mod(iloop, correlation_interval) == 0
                @printf "%11.6g" thiscor
                num_nan>0 && @printf "%s" string(" (", num_nan, " are NaN)")
            end
            println()
            flush(stdout);  flush(stderr)
        end
    catch e
        println("stopping early")
        showerror(stdout, e, stacktrace(catch_backtrace()))
        save_weights_P_raughist(string(iloop,'i'))
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
              :P => return_P_raughist && p.PCompute == :fast ? P : nothing,
              :raughist => return_P_raughist && p.PCompute == :small ? scratch.raughist : nothing)
end
