using NLsolve

function rate2synInput(p, ustd)
    targetRate_dict = load(parsed_args["spikerate_file"])
    targetRate = targetRate_dict[first(keys(targetRate_dict))]
    Ntime = floor(Int, (p.train_time-p.stim_off)/p.learn_every)
    if size(targetRate,1)!=Ntime
        error(parsed_args["spikerate_file"],
              " should have (train_time-stim_off)/learn_every = ",
              Ntime, " rows")
    end
    replace!(targetRate, 0.0=>0.1)
    xtarg = similar(targetRate)

    #---------- initial condition to Ricciardi ----------#
    initial_mu = 0.5*ones(Ntime)
    sigma = ustd / sqrt(p.tauedecay * 1.3) # factor 1.3 was calibrated manually
    invtau = 1000.0/p.taue
    VT = p.threshe
    Vr = p.vre
    F = Matrix{Float64}(undef, Ntime, Threads.nthreads())

    Threads.@threads for nid in 1:size(targetRate, 2)
        #---------- Solve Ricciardi ----------#
        xtarg[:,nid] .= solveRicci(F[:,Threads.threadid()],
                                   targetRate[:,nid], initial_mu, sigma, invtau, VT, Vr)
    end

    return xtarg
end

function solveRicci(F, rate, initial_mu, sigma, invtau, VT, Vr)
    sol = nlsolve((F,mu) -> f!(F,mu,sigma,invtau,VT,Vr,rate), initial_mu)
    return sol.zero
end

function f!(F, mu, sigma, invtau, VT, Vr, rate)
    F .= rate .- ricciardi.(mu, sigma, invtau, VT, Vr)
end

function ricciardi(mu,sigma,invtau,VT,Vr)
    # The function ricciardy calculates rate from the ricardi equation, with an error
    # less than 10^{-5}. If the IF neuron's voltage, V, satisfies between spikes
    # 
    #              tau dV/dt=mu-V_sigma*eta(t), 
    # 
    # where eta is the normalized white 
    # noise term, ricciardi calculates the firing rate.
    # It is called as
    # 
    #          rate=ricciardi(mu,sigma,invtau,VT,Vr)
    # 
    # where VT is the threshold voltage and Vr is the reset potential.
    
    VT<Vr && error("Threshold lower than reset in function ricciardi! I am quitting :(")
    sigma<0.0 && error("Negative  noise variance in function ricciardi! I am quitting :(")
    if sigma==0
        if mu<VT
            return 0
        else
            return invtau/log((mu-Vr)/(mu-VT))
        end
    end

    xp=(mu-Vr)/sigma;
    xm=(mu-VT)/sigma;
    if xm>0
        return invtau/(f_ricci(xp)-f_ricci(xm))
    elseif xp>0
        return invtau/(f_ricci(xp)+exp(xm^2)*g_ricci(-xm)) # relevant to the balanced network
    else
        return exp(-xm^2-log(g_ricci(-xm)-exp(xp^2-xm^2)*g_ricci(-xp)))*invtau
    end
    # f1=invtau/(f_ricci(xp)-f_ricci(xm))
    # f2=invtau/(f_ricci(xp)+exp(xm^2)*g_ricci(-xm))
    # f3=exp(-xm^2-log(g_ricci(-xm)-exp(xp^2-xm^2)*g_ricci(-xp)))*invtau
    # rate=f1*(xm>0)*(xp>0)+f2 *(xm<=0)*(xp>0)+f3*(xm<=0)*(xp<=0)
end

function f_ricci(x)
    z=x/(1+x)
    evalpoly(z, (log(2.0*x+1),
                 -2.2757881388024176*10^-1,
                 +7.7373949685442023*10^-1,
                 -3.2056016125642045*10^-1,
                 +3.2171431660633076*10^-1,
                 -6.2718906618071668*10^-1,
                 +9.3524391761244940*10^-1,
                 -1.0616084849547165,
                 +6.4290613877355551*10^-1,
                 -1.4805913578876898*10^-1))
end

function g_ricci(x)
    z=x/(2+x)
    enum = evalpoly(z, (0.0,
                        +3.5441754117462949,
                        -7.0529131065835378,
                        -5.6532378057580381*10^1,
                        +2.7956761105465944*10^2,
                        -5.2037554849441472*10^2,
                        +4.5658245777026514*10^2,
                        -1.5573340457809226*10^2))
    denom = evalpoly(z, (1,
                         -4.1357968834226053,
                         -7.2984226138266743,
                         +9.8656602235468327*10^1,
                         -3.3420436223415163*10^2,
                         +6.0108633903294185*10^2,
                         -5.9958577549598340*10^2,
                         +2.7718420330693891*10^2,
                         -1.6445022798669722*10^1))
    enum/denom
end
