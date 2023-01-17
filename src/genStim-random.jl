#=
the genStim plugin defines the external inputs added to the membrane voltage.
this file is the default, and returns for each neuron a random time-varying
signal.
=#

#=
return a T x Ncells matrix representing the inputs that trigger the learned
responses within the time interval [stim_on, stim_off]
=#

function genStim(args)
    @unpack stim_on, stim_off, dt, Ncells, mu, b, sig = args

    timeSteps = round(Int, (stim_off - stim_on) / dt)
    stim = zeros(timeSteps, Ncells)

    for ci = 1:Ncells
        for i = 1:timeSteps-1
            stim[i+1,ci] = stim[i,ci] +
                           b * (mu - stim[i,ci]) * dt +
                           sig * sqrt(dt) * randn(rng)
        end
    end

    return stim
end
