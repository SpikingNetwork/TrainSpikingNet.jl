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

    T = eltype(mu)

    timeSteps = round(Int, (stim_off - stim_on) / dt)
    stim = Array{T}(undef, timeSteps, Ncells)
    stim[1,:] .= T(0)

    Threads.@threads :static for ci=1:Ncells
        for i = 1:timeSteps-1
            stim[i+1,ci] = stim[i,ci] +
                           b * (mu - stim[i,ci]) * dt +
                           sig * sqrt(dt) * randn(copy_rng[Threads.threadid()])
        end
    end

    return stim
end
