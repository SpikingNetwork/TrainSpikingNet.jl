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

the vector (nc0) specifies the number of postsynaptic neurons
=#

function genStaticWeights(args)
    @unpack K, Ncells, Ne, pree, jee, jie, jei, jii, rng = args

    nc0Max = 2*K # outdegree
    nc0 = fill(nc0Max, Ncells)
    w0Index = zeros(Int, nc0Max, Ncells)
    w0Weights = zeros(nc0Max, Ncells)
    nc0Max > 0 && for i = 1:Ncells
        postcells = [1:i-1; i+1:Ncells]  # omit autapse
        w0Index[1:nc0Max,i] = sample(rng, postcells, nc0Max, replace=false, ordered=true) # fixed outdegree nc0Max
        nexc = count(w0Index[1:nc0Max,i] .<= Ne) # number of exc synapses
        if i <= Ne
            w0Weights[1:nexc,i] .= jee  ## EE weights
            w0Weights[nexc+1:nc0Max,i] .= jie  ## IE weights
        else
            w0Weights[1:nexc,i] .= jei  ## EI weights
            w0Weights[nexc+1:nc0Max,i] .= jii  ## II weights
        end
    end

    return w0Index, w0Weights, nc0
end
