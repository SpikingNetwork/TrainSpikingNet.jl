function genInitialWeights(p)

nc0Max = round(Int, p.Ncells*p.pree) # outdegree
nc0 = fill(nc0Max, p.Ncells)
w0Index = zeros(Int,nc0Max,p.Ncells)
w0Weights = zeros(nc0Max,p.Ncells)
nc0Max > 0 && for i = 1:p.Ncells
    postcells = [1:i-1; i+1:p.Ncells]  # omit autapse
    w0Index[1:nc0Max,i] = sample(postcells, nc0Max, replace=false, ordered=true) # fixed outdegree nc0Max
    nexc = count(w0Index[1:nc0Max,i] .<= p.Ne) # number of exc synapses
    if i <= p.Ne
        w0Weights[1:nexc,i] .= p.jee  ## EE weights
        w0Weights[nexc+1:nc0Max,i] .= p.jie  ## IE weights
    else
        w0Weights[1:nexc,i] .= p.jei  ## EI weights
        w0Weights[nexc+1:nc0Max,i] .= p.jii  ## II weights
    end
end

return w0Index, w0Weights, nc0

end
