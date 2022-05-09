function genffwdRate(p)
    Nsteps = round(Int, (p.train_time - p.stim_off) / p.dt)
    ffwdRate = p.ffwdRate_mu*ones(Nsteps, p.Lffwd)

    for i = 1:Nsteps-1
        ffwdRate[i+1,:] = ffwdRate[i,:] + p.ffwdRate_bou*(p.ffwdRate_mu .- ffwdRate[i,:])*p.dt + p.ffwdRate_sig*sqrt(p.dt)*randn(rng, p.Lffwd);
    end

    ffwdRate = funMovAvg(ffwdRate, 500)
    clamp!(ffwdRate, 0, Inf)
end


function funMovAvg(x,wid)
    Nsteps = size(x,1)
    movavg = similar(x)
    for i = 1:Nsteps
        Lind = maximum([i-wid, 1])
        Rind = minimum([i+wid, Nsteps])
        xslice = @view x[Lind:Rind,:]
        movavg[i,:] = mean(xslice, dims=1)
    end

    return movavg
end
