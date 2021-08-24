function genTarget(p,uavg,biasType)

sampled_Nsteps = Int((p.train_time - p.stim_off)/p.learn_every)
utargSampled = Array{Float64}(undef, sampled_Nsteps, p.Ncells)

time = collect(1:p.Nsteps)*p.dt
bias = Array{Float64}(undef, p.Nsteps)


#----- ZERO ----#
if biasType == "zero"
    bias .= 0
end

#----- OU ----#
if biasType == "ou"
    mu_ou_bias = 0.0;
    b_ou_bias = 1/400;
    sig_ou_bias = 0.02;
    bias[1] = 0
    for i = 1:p.Nsteps-1
        bias[i+1] = bias[i]+b_ou_bias*(mu_ou_bias-bias[i])*p.dt + sig_ou_bias*sqrt(p.dt)*randn();
    end
end

#----- RAMPING ----#
if biasType == "ramping"
    Nstart = round(Int, p.stim_off/p.dt)
    bias[1:Nstart-1] .= 0
    bias[Nstart:p.Nsteps] = 0.25/(p.Nsteps-Nstart)*collect(0:p.Nsteps-Nstart)
    bias[p.Nsteps+1:end] .= 0
end

for j=1:p.Ncells
    A  = 0.5
    period = 1000.0;
    phase = period*rand();
    fluc = A*sin.((time.-phase).*(2*pi/period)) .+ uavg[j];
    utargSampled[:,j] = funSample(p,fluc + bias)
end

return utargSampled 

end
