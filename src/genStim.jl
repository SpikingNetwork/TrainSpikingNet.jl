function genStim(p)

    timeSteps = round(Int, (p.stim_off - p.stim_on)/p.dt)
    stim = zeros(timeSteps,p.Ncells)
    mu = 0.0;
    b = 1/20;
    sig = 0.2;
    for ci = 1:p.Ncells
        for i = 1:timeSteps-1
            stim[i+1,ci] = stim[i,ci]+b*(mu-stim[i,ci])*p.dt + sig*sqrt(p.dt)*randn(rng);
        end
    end

    return stim

end
