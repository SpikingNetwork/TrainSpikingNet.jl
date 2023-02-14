#=
the genPlasticWeights plugin defines the connectivity of the learned synapses.
this file is the default, and for each neuron simply chooses a random Lexc + Linh
as presynaptic partners and sets their initial weights to predefined values.
=#

#=
this is the sole plugin to input a variable other than just the user-defined
arguments.  specifically, it also inputs the number of spikes generated by the
static connectivity alone (called ns0 in the code).

returned are three matrices and a vector specifying the plastic connectivity.

the first returned matrix (called wpWeightX in the source code) is
Ncells x LX and contains the (initial) weights of the feed forward
presynaptic neurons.

the second (wpWeightIn, Ncells columns) and third (wpIndexIn, Ncells columns)
matrices contain the weights and indices of the recurrent presynaptic
neurons.

the final returned variable is a vector of length Ncells (ncpIn) which
specifies how many presynaptic connections each neuron has, or equivalently,
the length of each ragged column in the two matrices.  elements beyond the
ragged edge in wpIndexIn must by zero.
=#

function genPlasticWeights(args, ns0)
    @unpack Ncells, frac, Ne, Lexc, Linh, LX, wpee, wpie, wpei, wpii, wpX, rng, seed = args

    num_threads = Threads.nthreads()
    copy_rng = [typeof(rng)() for _=1:num_threads];
    isnothing(seed) || Random.seed!.(copy_rng, seed .+ (1:num_threads))
    save(joinpath(data_dir,"rng-genPlasticWeights.jld2"), "rng", copy_rng)

    # order neurons by their firing rate
    frac_cells = round(Int, frac*Ne)
    exc_ns0 = ns0[1:Ne]
    inh_ns0 = ns0[Ne+1:Ncells]
    exc_ordered = sortperm(exc_ns0)
    inh_ordered = collect(Ne+1:Ncells)[sortperm(inh_ns0)]
    exc_selected = sort(exc_ordered[end-frac_cells+1:end])
    inh_selected = sort(inh_ordered[end-frac_cells+1:end])
    
    # define weights_plastic
    wpWeightIn = Array{Float64}(undef, Lexc+Linh, Ncells)
    wpIndexIn = Array{Int}(undef, Lexc+Linh, Ncells)
    ncpIn = Array{Int}(undef, Ncells)

    # select random exc and inh presynaptic neurons
    function random_plastic_recurrent(I, tid)
        for i in I
            # (1) select consecutive neurons from a random starting point
            # rnd_start = rand(1:length(exc_selected)-Lexc-Linh+1)
            # indE = sort(exc_selected[rnd_start:rnd_start+Lexc-1])
            # indI = sort(inh_selected[rnd_start:rnd_start+Linh-1])

            # (2) select random neurons
            indE = sample(copy_rng[tid], exc_selected, Lexc, replace=false)
            indI = sample(copy_rng[tid], inh_selected, Linh, replace=false)

            # build wpIndexIn
            wpIndexIn[1:Lexc, i] = indE
            wpIndexIn[Lexc+1:end, i] = indI
            ncpIn[i] = Lexc+Linh

            # initial exc and inh plastic weights
            if i <= Ne
                wpWeightIn[1:Lexc, i] .= wpee
                wpWeightIn[Lexc+1:end, i] .= wpei
            else
                wpWeightIn[1:Lexc, i] .= wpie
                wpWeightIn[Lexc+1:end, i] .= wpii
            end
        end
    end
    tasks = Vector{Task}(undef, num_threads)
    partitions = [floor.(Int, collect(1:(Ncells/num_threads):Ncells)); Ncells+1]
    for i=1:num_threads
        # Threads.@threads does NOT guarantee a particular threadid for each partition
        # so the RNG seed might be different
        tasks[i] = Threads.@spawn random_plastic_recurrent(partitions[i]:partitions[i+1]-1, i)
    end
    for i = 1:num_threads
        wait(tasks[i])
    end

    # define feedforward weights to all neurons
    #       - wpWeightX = randn(Ncells, LX) * wpX
    #       - initial weights, wpX = 0
    wpWeightX = randn(copy_rng[1], Ncells, LX) * wpX
    
    return wpWeightX, wpWeightIn, wpIndexIn, ncpIn
end
