function funSample(p,X)

    timev = collect(1:p.Nsteps)*p.dt
    idx = timev .>= p.stim_off + p.learn_every

    if ndims(X) == 1
        XpostStim = X[idx]
        Xsampled = XpostStim[1:p.learn_step:end]
    elseif ndims(X) == 2
        XpostStim = X[idx,:]
        Xsampled = XpostStim[1:p.learn_step:end,:]
    end

    return Xsampled

end