# return a T x Lffwd matrix spiking thresholds of the feed forward neurons

function genFfwdRate(args)
    @unpack train_time, stim_off, dt, Lffwd, mu, bou, sig, wid = args

    Nsteps = round(Int, (train_time - stim_off) / dt)
    ffwdRate = mu*ones(Nsteps, Lffwd)

    for i = 1:Nsteps-1
        ffwdRate[i+1,:] = ffwdRate[i,:] + bou*(mu .- ffwdRate[i,:])*dt + sig*sqrt(dt)*randn(rng, Lffwd);
    end

    ffwdRate = funMovAvg(ffwdRate, wid)
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
