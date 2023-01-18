#=
the genFfwdRate plugin defines the threshold for the Poisson feed-forward neurons.
this file is the default, and for each neuron simply chooses a random Lffwd as
presynaptic partners and sets their initial weights to predefined values.
=#

#=
return a single Nsteps x Lffwd matrix of time-varying spiking thresholds
for the feed forward neurons
=#

function genFfwdRate(args)
    @unpack train_time, stim_off, dt, Lffwd, mu, bou, sig, wid, rng = args

    Nsteps = round(Int, (train_time - stim_off) / dt)
    ffwdRate = Array{Float64}(undef, Nsteps, Lffwd)
    ffwdRate[1,:] .= mu

    for i = 1:Nsteps-1
        ffwdRate[i+1,:] = ffwdRate[i,:] +
                          bou * (mu .- ffwdRate[i,:]) * dt + 
                          sig * sqrt(dt) * randn(rng, Lffwd);
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
