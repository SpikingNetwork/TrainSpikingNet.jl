function genPlasticWeights(p, w0Index, nc0, ns0)

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

    # order neurons by their firing rate
    frac_neurons_selected = p.frac
    frac_cells = Int(frac_neurons_selected*p.Ne)
    exc_ns0 = ns0[1:p.Ne]
    inh_ns0 = ns0[p.Ne+1:p.Ncells]
    exc_ordered = sortperm(exc_ns0)
    inh_ordered = collect(p.Ne+1:p.Ncells)[sortperm(inh_ns0)]
    exc_selected = exc_ordered[end-frac_cells+1:end]
    inh_selected = inh_ordered[end-frac_cells+1:end]
    
    # define weights_plastic
    wpWeightIn = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    wpIndexIn = zeros(p.Ncells,round(Int,p.Lexc+p.Linh))
    ncpIn = zeros(Int,p.Ncells)
    for postCell = 1:p.Ncells
        # select random exc and inh presynaptic neurons
        # # (1) select consecutive neurons
        # rnd_start = rand(1:length(exc_selected)-p.L+1)
        # indE = sort(exc_selected[rnd_start:rnd_start+p.L-1])
        # indI = sort(inh_selected[rnd_start:rnd_start+p.L-1])

        # (2) select random neurons
        indE = sort(shuffle(exc_selected)[1:p.L])
        indI = sort(shuffle(inh_selected)[1:p.L])

        # build wpIndexIn
        ind  = [indE; indI]
        wpIndexIn[postCell,:] = ind
        ncpIn[postCell] = length(ind)

        # initial exc and inh plastic weights
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
    
    return wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut
    
end
