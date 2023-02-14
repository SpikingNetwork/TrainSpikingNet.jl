#=
the genStaticWeights plugin defines the connectivity and weights of the
fixed synapses.  this file is the default, and for each neuron simply
pools the other neurons, chooses a random 2*K as presynaptic partners, and
sets their weights to predefined values.
=#

#=
returned are two matrices each with Ncells columns and a vector of length
Ncells specifying the static connectivity.

the first returned matrix (called w0Index in the source code) contains the
index of the postsynaptic neurons and the second (w0Weights) contains the
synaptic weights.

the vector (nc0) specifies the number of postsynaptic neurons, or
equivalently, the length of each ragged column in the two matrices.
elements beyond the ragged edge in w0Index must by zero.
=#

skip_autapse(a, i) = i<a ? i : i+1

function genStaticWeights(args)
    @unpack K, Ncells, Ne, jee, jie, jei, jii, rng, seed = args

    num_threads = Threads.nthreads()
    copy_rng = [typeof(rng)() for _=1:num_threads];
    isnothing(seed) || Random.seed!.(copy_rng, seed .+ (1:num_threads))
    save(joinpath(data_dir,"rng-genStaticWeights.jld2"), "rng", copy_rng)

    nc0Max = 2*K  # outdegree
    nc0 = fill(nc0Max, Ncells)
    w0Index = zeros(Int, nc0Max, Ncells)
    w0Weights = zeros(nc0Max, Ncells)
    postcells = 1:Ncells-1

    nc0Max == 0 && return w0Index, w0Weights, nc0

    function random_static_recurrent(I, tid)
        for i in I
            w0Index[1:nc0Max,i] = skip_autapse.(i, sample(copy_rng[tid], postcells, nc0Max, replace=false)) # fixed outdegree nc0Max
            if i <= Ne
                @views w0Weights[:,i] .= ifelse.(w0Index[:,i].<=Ne, jee, jie)
            else
                @views w0Weights[:,i] .= ifelse.(w0Index[:,i].<=Ne, jei, jii)
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

    return w0Index, w0Weights, nc0
end
