#=
the genUTarget plugin generates the target synaptic currents.  it can do
so either algorithmically, or by simply loading data from a file.  this
file is the default, and for each neuron generates a sinusiod with a
random phase.
=#

#=
return a T x Ncells matrix representing the desired synaptic currents to
be learned
=#

function genUTarget(args, uavg)
    @unpack train_time, stim_off, learn_every, Ncells, Nsteps, dt, A, period, biasType, mu_ou_bias, b_ou_bias, sig_ou_bias, rng, seed = args

    num_threads = Threads.nthreads()
    copy_rng = [typeof(rng)() for _=1:num_threads];
    isnothing(seed) || Random.seed!.(copy_rng, seed .+ (1:num_threads))
    save(joinpath(data_dir,"rng-genUTarget.jld2"), "rng", copy_rng)

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
                        sig_ou_bias * sqrt(dt) * randn(copy_rng[1])
        end
    elseif biasType == :ramping
        Nstart = round(Int, stim_off/dt)
        bias[1:Nstart-1] .= 0
        bias[Nstart:Nsteps] = 0.25 / (Nsteps-Nstart) * [0:Nsteps-Nstart]
        bias[Nsteps+1:end] .= 0
    else
        error("biasType must be one of :zero, :ou, or :ramping")
    end

    idx = time .>= stim_off + learn_every
    time = time[idx][1:learn_step:end]
    bias = bias[idx][1:learn_step:end]
    time .*= 2*pi / period

    function random_phase_sinusoids(I, tid)
        for i in I
            phase = period * rand(copy_rng[tid])
            utargSampled[:,i] = A * sin.(time.-phase) .+ uavg[i] .+ bias
        end
    end
    tasks = Vector{Task}(undef, num_threads)
    partitions = [floor.(Int, collect(1:(Ncells/num_threads):Ncells)); Ncells+1]
    for i=1:num_threads
        # Threads.@threads does NOT guarantee a particular threadid for each partition
        # so the RNG seed might be different
        tasks[i] = Threads.@spawn random_phase_sinusoids(partitions[i]:partitions[i+1]-1, i)
    end
    for i = 1:num_threads
        wait(tasks[i])
    end

    return utargSampled 
end
