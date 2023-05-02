#=
the genStaticWeights plugin defines the connectivity and weights of the
fixed synapses.  this file is the default, and for each neuron simply
pools the other neurons, chooses a random 2*K as presynaptic partners, and
sets their weights to predefined values.
=#

#=
returned are two vectors of vectors specifying the static connectivity.

the first returned vector (called w0Index in the source code) is of length
Ncells and contains a vector for each neuron of the indices of its postsynaptic
neurons, and the second (w0Weights) contains the corresponding synaptic weights.
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
    w0Weights = Vector{Vector{eltype(jee)}}(undef, Ncells)
    postcells = 1:Ncells-1

    nc0Max == 0 && return w0Index, w0Weights

    Threads.@threads :static for i=1:Ncells
        w0Index[i] = skip_autapse.(i, sample(copy_rng[Threads.threadid()], postcells, nc0Max, replace=false)) # fixed outdegree nc0Max
        w0Weights[i] = Array{eltype(jee)}(undef, nc0Max)
        if i <= Ne
            @views w0Weights[i] .= ifelse.(w0Index[i].<=Ne, jee, jie)
        else
            @views w0Weights[i] .= ifelse.(w0Index[i].<=Ne, jei, jii)
        end
    end

    return w0Index, w0Weights
end
