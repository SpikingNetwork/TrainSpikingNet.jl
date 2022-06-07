# return a T x Ncells matrix representing the desired currents that are learned

function genTarget(args, uavg)
    train_time, stim_off, learn_every, Ncells, Nsteps, dt, A, period, biasType, mu_ou_bias, b_ou_bias, sig_ou_bias = map(x->args[x],
         [:train_time, :stim_off, :learn_every, :Ncells, :Nsteps, :dt, :A, :period, :biasType, :mu_ou_bias, :b_ou_bias, :sig_ou_bias])

    sampled_Nsteps = round(Int, (p.train_time - p.stim_off) / p.learn_every)
    utargSampled = Array{Float64}(undef, sampled_Nsteps, p.Ncells)

    time = collect(1:p.Nsteps)*p.dt
    bias = Array{Float64}(undef, p.Nsteps)


    #----- ZERO ----#
    if biasType == "zero"
        bias .= 0
    end

    #----- OU ----#
    if biasType == "ou"
        bias[1] = 0
        for i = 1:p.Nsteps-1
            bias[i+1] = bias[i] + b_ou_bias*(mu_ou_bias-bias[i])*p.dt + sig_ou_bias*sqrt(p.dt)*randn(rng)
        end
    end

    #----- RAMPING ----#
    if biasType == "ramping"
        Nstart = round(Int, p.stim_off/p.dt)
        bias[1:Nstart-1] .= 0
        bias[Nstart:p.Nsteps] = 0.25/(p.Nsteps-Nstart)*collect(0:p.Nsteps-Nstart)
        bias[p.Nsteps+1:end] .= 0
    end

    for j=1:p.Ncells
        phase = period*rand(rng)
        fluc = A*sin.((time.-phase).*(2*pi/period)) .+ uavg[j]
        utargSampled[:,j] = funSample(Nsteps, dt, stim_off, learn_every, fluc + bias)
    end

    return utargSampled 
end


function funSample(Nsteps, dt, stim_off, learn_every, X)
    timev = collect(1:Nsteps)*dt
    idx = timev .>= stim_off + learn_every

    if ndims(X) == 1
        XpostStim = X[idx]
        Xsampled = XpostStim[1:learn_step:end]
    elseif ndims(X) == 2
        XpostStim = X[idx,:]
        Xsampled = XpostStim[1:learn_step:end,:]
    end

    return Xsampled
end
