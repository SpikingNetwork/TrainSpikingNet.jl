function genStim(p)

# stim = 0.5*(2*rand(p.Ncells) .- 1)
timeSteps = Int((p.stim_off - p.stim_on)/p.dt)
stim = zeros(timeSteps,p.Ncells)
mu = 0.0;
b = 1/20;
sig = 0.2; #0.1;
for ci = 1:Ncells
    for i = 1:timeSteps-1
        stim[i+1,ci] = stim[i,ci]+b*(mu-stim[i,ci])*dt + sig*sqrt(dt)*randn();
    end
end

return stim

end