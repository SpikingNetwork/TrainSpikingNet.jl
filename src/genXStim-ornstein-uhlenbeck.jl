#=
the genXStim plugin defines the external inputs added to the membrane voltage.
this file is the default, and returns for each neuron a random time-varying
signal.
=#

#=
return a T x Ncells matrix representing the inputs that trigger the learned
responses within the time interval [stim_on, stim_off]
=#

function genXStim(args)
    @unpack stim_on, stim_off, dt, Ncells, mu, b, sig, rng, seed = args

    num_threads = Threads.nthreads()
    copy_rng = [typeof(rng)() for _=1:num_threads];
    isnothing(seed) || Random.seed!.(copy_rng, seed .+ (1:num_threads))
    save(joinpath(data_dir,"rng-genUTarget.jld2"), "rng", copy_rng)

    timeSteps = round(Int, (stim_off - stim_on) / dt)
    stim = Array{Float64}(undef, timeSteps, Ncells)
    stim[1,:] .= 0

    function random_external_currents(I, tid)
        for ci in I
            for i = 1:timeSteps-1
                stim[i+1,ci] = stim[i,ci] +
                               b * (mu - stim[i,ci]) * dt +
                               sig * sqrt(dt) * randn(copy_rng[tid])
            end
        end
    end
    tasks = Vector{Task}(undef, num_threads)
    partitions = [floor.(Int, collect(1:(Ncells/num_threads):Ncells)); Ncells+1]
    for i=1:num_threads
        # Threads.@threads does NOT guarantee a particular threadid for each partition
        # so the RNG seed might be different
        tasks[i] = Threads.@spawn random_external_currents(partitions[i]:partitions[i+1]-1, i)
    end
    for i = 1:num_threads
        wait(tasks[i])
    end

    return stim
end
