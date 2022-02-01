function ricciardi(mu,sigma,tau,VT,Vr)
    # The function ricciardy calculates rate from the ricardi equation, with an error
    # less than 10^{-5}. If the IF neuron's voltage, V, satisfies between spikes
    # 
    #              tau dV/dt=mu-V_sigma*eta(t), 
    # 
    # where eta is the normalized white 
    # noise term, ricciardi calculates the firing rate.
    # It is called as
    # 
    #          rate=ricciardi(mu,sigma,tau,VT,Vr)
    # 
    # where VT is the threshold voltage and Vr is the reset potential.
    
    if VT<Vr
        println("Threshold lower than reset in function ricciardi! I am quitting :(")
    end
    if sigma<0.0
        println("Negative  noise variance in function ricciardi! I am quitting :(")
    end
    if sigma==0
        if mu<VT
            rate=0
            return rate
        else
            rate=1.0/(tau.*log((mu-Vr)./(mu-VT)))
            return rate
        end
    end

    xp=(mu-Vr)./sigma;
    xm=(mu-VT)./sigma;
    if xm>0
        rate=1.0./tau./(f_ricci(xp)-f_ricci(xm));
    elseif xp>0
        rate=1.0./tau./(f_ricci(xp)+exp(xm.^2).*g_ricci(-xm)); # relevant to the balanced network
    else
        rate=exp(-xm.^2-log(g_ricci(-xm)-exp(xp.^2-xm.^2).*g_ricci(-xp)))./tau;
    end
    # f1=1.0/tau./(f_ricci(xp)-f_ricci(xm));
    # f2=1.0/tau./(f_ricci(xp)+exp(xm.^2).*g_ricci(-xm)) ;
    # f3=exp(-xm.^2-log(g_ricci(-xm)-exp(xp.^2-xm.^2).*g_ricci(-xp)))./tau;
    # rate=f1.*(xm>0).*(xp>0)+f2 .*(xm<=0).*(xp>0)+f3.*(xm<=0).*(xp<=0);
        
    return rate

end

function f_ricci(x)
    z=x./(1+x)
    f_ricci=log(2.0*x+1) - 2.2757881388024176*10^-1*z + 7.7373949685442023*10^-1*z.^2 - 3.2056016125642045*10^-1*z.^3 + 3.2171431660633076*10^-1*z.^4 - 6.2718906618071668*10^-1*z.^5 + 9.3524391761244940*10^-1*z.^6 - 1.0616084849547165 *z.^7 + 6.4290613877355551*10^-1*z.^8 - 1.4805913578876898*10^-1*z.^9
    return f_ricci
end

function g_ricci(x)
    z=x./(2+x)
    enum=3.5441754117462949*z - 7.0529131065835378*z.^2 - 5.6532378057580381*10^1*z.^3 + 2.7956761105465944*10^2*z.^4 - 5.2037554849441472*10^2*z.^5 + 4.5658245777026514*10^2*z.^6 - 1.5573340457809226*10^2*z.^7
    denom=1-4.1357968834226053*z - 7.2984226138266743*z.^2 + 9.8656602235468327*10^1*z.^3 - 3.3420436223415163*10^2*z.^4 + 6.0108633903294185*10^2*z.^5 - 5.9958577549598340*10^2*z.^6 + 2.7718420330693891*10^2*z.^7 - 1.6445022798669722*10^1*z.^8
    g_ricci=enum./denom
    return g_ricci
end