#=
the genRateX plugin defines the threshold for the Poisson feed-forward neurons.
this file is the default, and for each neuron simply chooses a random LX as
presynaptic partners and sets their initial weights to predefined values.
=#

#=
return a single Nsteps x LX matrix of time-varying spiking thresholds
for the feed forward neurons
=#

function genRateX(args)
    @unpack train_time, stim_off, dt, LX, mu, bou, sig, wid, rng = args

    Nsteps = round(Int, (train_time - stim_off) / dt)
    rateX = Array{Float64}(undef, Nsteps, LX)
    rateX[1,:] .= mu

    for i = 1:Nsteps-1
        rateX[i+1,:] = rateX[i,:] +
                          bou * (mu .- rateX[i,:]) * dt + 
                          sig * sqrt(dt) * randn(rng, LX);
    end

    rateX = funMovAvg(rateX, wid)
    clamp!(rateX, 0, Inf)
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
