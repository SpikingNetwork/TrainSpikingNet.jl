#=
the genStaticWeights plugin defines the connectivity and weights of the
fixed synapses.  this file is the default, and for each neuron simply
pools the other neurons, chooses a random 2*K as presynaptic partners, and
sets their weights to predefined values.
=#

#=
returned are two vectors of vectors specifying the static connectivity.

the first returned vector (called w0Index in the source code) is of length Ncells
and contains the
index of the postsynaptic neurons and the second (w0Weights) contains the
synaptic weights.

=#

skip_autapse(a, i) = i<a ? i : i+1

function genStaticWeights(args)
    @unpack K, Ncells, Ne, jee, jie, jei, jii, rng, seed = args

    num_threads = Threads.nthreads()
    copy_rng = [typeof(rng)() for _=1:num_threads];
    isnothing(seed) || Random.seed!.(copy_rng, seed .+ (1:num_threads))
    save(joinpath(data_dir,"rng-genStaticWeights.jld2"), "rng", copy_rng)

    nc0Max = 2*K  # outdegree
    w0Index = Vector{Vector{Int}}(undef, Ncells)
    w0Weights = Vector{Vector{Float64}}(undef, Ncells)
    postcells = 1:Ncells-1

    nc0Max == 0 && return w0Index, w0Weights

    function random_static_recurrent(I, tid)
        for i in I
            w0Index[i] = skip_autapse.(i, sample(copy_rng[tid], postcells, nc0Max, replace=false)) # fixed outdegree nc0Max
            w0Weights[i] = Array{Float64}(undef, nc0Max)
            if i <= Ne
                @views w0Weights[i] .= ifelse.(w0Index[i].<=Ne, jee, jie)
            else
                @views w0Weights[i] .= ifelse.(w0Index[i].<=Ne, jei, jii)
            end
        end
    end
    tasks = Vector{Task}(undef, num_threads)
    partitions = [floor.(Int, collect(1:(Ncells/num_threads):Ncells)); Ncells+1]
    for i=1:num_threads
        # Threads.@threads does NOT guarantee a particular threadid for each partition
        # so the RNG seed might be different
        tasks[i] = Threads.@spawn random_static_recurrent(partitions[i]:partitions[i+1]-1, i)
    end
    for i = 1:num_threads
        wait(tasks[i])
    end

    return w0Index, w0Weights
end
