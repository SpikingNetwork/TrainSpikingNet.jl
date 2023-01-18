#=
the genTarget plugin generates the target synaptic currents.  it can do
so either algorithmically, or by simply loading data from a file.  this
file is the default, and for each neuron generates a sinusiod with a
random phase.
=#

#=
return a T x Ncells matrix representing the desired synaptic currents to
be learned
=#

function genTarget(args, uavg)
    @unpack train_time, stim_off, learn_every, Ncells, Nsteps, dt, A, period, biasType, mu_ou_bias, b_ou_bias, sig_ou_bias, rng = args

    sampled_Nsteps = round(Int, (train_time - stim_off) / learn_every)
    utargSampled = Array{Float64}(undef, sampled_Nsteps, Ncells)

    time = collect(1:Nsteps)*dt
    bias = Array{Float64}(undef, Nsteps)


    if biasType == :zero
        bias .= 0
    elseif biasType == :ou
        bias[1] = 0
        for i = 1:Nsteps-1
            bias[i+1] = bias[i] +
                        b_ou_bias * (mu_ou_bias - bias[i]) * dt +
                        sig_ou_bias * sqrt(dt) * randn(rng)
        end
    elseif biasType == :ramping
        Nstart = round(Int, stim_off/dt)
        bias[1:Nstart-1] .= 0
        bias[Nstart:Nsteps] = 0.25/(Nsteps-Nstart)*collect(0:Nsteps-Nstart)
        bias[Nsteps+1:end] .= 0
    else
        error("biasType must be one of :zero, :ou, or :ramping")
    end

    for j=1:Ncells
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
