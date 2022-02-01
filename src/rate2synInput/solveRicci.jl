function solveRicci(rate, initial_mu, sigma, tau, VT, Vr)
    sol = nlsolve((F,mu) -> f!(F,mu,sigma,tau,VT,Vr,rate), initial_mu)
    return sol.zero
end

function f!(F, mu, sigma, tau, VT, Vr, rate)
    F .= rate .- ricciardi.(mu, sigma, tau, VT, Vr)
end
