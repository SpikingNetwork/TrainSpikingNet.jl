function genTarget(p,uavg,biasType)

Nsteps = p.Nsteps
Ncells = p.Ncells
utarg  = zeros(Nsteps,Ncells)
fluc  = zeros(Nsteps,Ncells)

sampled_Nsteps = Int((p.train_time - p.stim_off)/p.learn_every)
utargSampled = zeros(sampled_Nsteps,Ncells) #funSample(p,utarg)
flucSampled = zeros(sampled_Nsteps,Ncells)
biasSampled = zeros(sampled_Nsteps)

time = collect(1:Nsteps)*p.dt
bias = zeros(Nsteps)
phases = zeros(Ncells)


#----- ZERO ----#
if biasType == "zero"
    bias = zeros(Nsteps)
end

#----- OU ----#
if biasType == "ou"
    mu_ou_bias = 0.0;
    b_ou_bias = 1/400;
    sig_ou_bias = 0.02; #0.02 works
    for i = 1:Nsteps-1
        bias[i+1] = bias[i]+b_ou_bias*(mu_ou_bias-bias[i])*p.dt + sig_ou_bias*sqrt(p.dt)*randn();
    end
end

#----- RAMPING ----#
if biasType == "ramping"
    Nstart = Int(p.stim_off/p.dt)
    bias = zeros(Nsteps)
    bias[Nstart:Nsteps] = 0.25/(Nsteps-Nstart)*collect(0:Nsteps-Nstart)
end

for j=1:Ncells
    A  = 0.5 # default 0.5
    period = 1000.0;
    phase = period*rand();
    phases[j] = phase
    fluc = A*sin.((time.-phase).*(2*pi/period)) .+ uavg[j];
    utargSampled[:,j] = funSample(p,fluc + bias)
    flucSampled[:,j] = funSample(p,fluc)
end
biasSampled = funSample(p,bias)

return utargSampled 



end