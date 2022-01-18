function genPlasticWeights(p, w0Index, nc0, ns0, matchedCells, trainEI)

    # rearrange initial weights
    w0 = Dict{Int,Array{Int,1}}()
    for i = 1:p.Ncells
        w0[i] = []
    end
    for preCell = 1:p.Ncells
        for i = 1:nc0[preCell]
            postCell = w0Index[i,preCell]
            push!(w0[postCell],preCell)
        end
    end

    # exc / inh neuron index
    exc_selected = collect(1:p.Ne)
    inh_selected = collect(p.Ne+1:p.Ncells)

    # define weights_plastic    
    wpWeightIn = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    wpIndexIn = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    ncpIn = zeros(Int,p.Ncells)

    #----------------------------------------------------------------------------#
    # (1) Set up (dummy) plastic connections from random presynaptic neurons, wpIndexIn
    #       - plastic weights, wpWeightIn, are set to the default value, 0.
    #----------------------------------------------------------------------------#
    for postCell = 1:p.Ncells
        # select random neurons
        indE = sort(shuffle(exc_selected)[1:p.L])
        indI = sort(shuffle(inh_selected)[1:p.L])

        # build wpIndexIn
        ind = [indE; indI]
        wpIndexIn[postCell,:] = ind
        ncpIn[postCell] = length(ind)
    end

    ##########################################################
    # update wpIndexIn, wpWeightIn for the matchedCells only
    #       - plastic weights to the untrained neurons = 0 
    ##########################################################

    #----------------------------------------------------------------------------#
    # (2a) If exc neurons trained, 
    #       - exc plastic connections come from other trained exc neurons.
    #       - inh plastic connections come from random inh neurons.
    # (2b) If inh neurons trained, 
    #       - exc plastic connections come from random exc neurons.
    #       - inh plastic connections come from other trained inh neurons.
    #----------------------------------------------------------------------------#
    for ii = 1:length(matchedCells)
        # neuron to be trained
        postCell = matchedCells[ii]

        #----------------------------------------------------#
        # select presynaptic neurons to the matchedCells
        #----------------------------------------------------#
        if trainEI == "exc"
            matchedCells_noautapse = filter(x->x!=postCell, matchedCells)
            indE = sort(shuffle(matchedCells_noautapse)[1:p.L])
            indI = sort(shuffle(inh_selected)[1:p.L])
        elseif trainEI == "inh"
            matchedCells_noautapse = filter(x->x!=postCell, matchedCells)
            indE = sort(shuffle(exc_selected)[1:p.L])
            indI = sort(shuffle(matchedCells_noautapse)[1:p.L])
        end

        # updated wpIndexIn for postcell in matchedCells
        ind = [indE; indI]
        wpIndexIn[postCell,:] = ind
        ncpIn[postCell] = length(ind)

        #----------------------------------------------------#
        # Update plastic weights to matchedCells
        #   - other plastic weights to untrained neurons = 0 
        #----------------------------------------------------#
        if postCell <= p.Ne
            wpee = p.wpee*ones(p.Lexc)
            wpei = p.wpei*ones(p.Linh)
            wpWeightIn[postCell,:] = [wpee; wpei]
        else
            wpie = p.wpie*ones(p.Lexc)
            wpii = p.wpii*ones(p.Linh)
            wpWeightIn[postCell,:] = [wpie; wpii]
        end
    end
    
    #####################################################
    ################ Feedforward inputs #################
    #####################################################
    #---------------------------------------------------#
    # define feedforward weights to all neurons
    #       - wpWeightFfwd = randn(p.Ncells, p.Lffwd) * p.wpffwd
    #       - initial weights, p.wpffwd = 0
    #---------------------------------------------------#
    wpWeightFfwd = Vector{Array{Float64,2}}(); 
    for licki = 1:2
        wtmp = randn(p.Ncells, p.Lffwd) * p.wpffwd
        push!(wpWeightFfwd, wtmp)
    end

    # get indices of postsynaptic cells for each presynaptic cell
    wpIndexConvert = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    wpIndexOutD = Dict{Int,Array{Int,1}}()
    ncpOut = zeros(Int,p.Ncells)
    for i = 1:p.Ncells
        wpIndexOutD[i] = []
    end
    for postCell = 1:p.Ncells
        for i = 1:ncpIn[postCell]
            preCell = wpIndexIn[postCell,i]
            push!(wpIndexOutD[preCell],postCell)
            wpIndexConvert[postCell,i] = length(wpIndexOutD[preCell])
        end
    end
    for preCell = 1:p.Ncells
        ncpOut[preCell] = length(wpIndexOutD[preCell])
    end

    # get weight, index of outgoing connections
    ncpOutMax = Int(maximum(ncpOut))
    wpIndexOut = zeros(ncpOutMax,p.Ncells)
    wpWeightOut = zeros(ncpOutMax,p.Ncells)
    for preCell = 1:p.Ncells
        wpIndexOut[1:ncpOut[preCell],preCell] = wpIndexOutD[preCell]
    end
    wpWeightOut = convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    
    return wpWeightFfwd, wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut
    
end
