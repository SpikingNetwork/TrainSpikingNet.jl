function genInitialWeights(p)

nc0Max = round(Int, p.Ncells*p.pree) # outdegree
nc0 = fill(nc0Max, p.Ncells)
w0Index = zeros(Int,nc0Max,p.Ncells)
w0Weights = zeros(nc0Max,p.Ncells)
for i = 1:p.Ncells
    postcells = filter(x->x!=i, collect(1:p.Ncells)) # remove autapse
    w0Index[1:nc0Max,i] = sort(shuffle(postcells)[1:nc0Max]) # fixed outdegree nc0Max
    nexc = sum(w0Index[1:nc0Max,i] .<= p.Ne) # number of exc synapses
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
