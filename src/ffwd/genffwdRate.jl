function genffwdRate(p, ffwdRate_mean)

# ffwdRate_mean = ffwdRate_list[mm]

dt = 0.1
numFfwd = p.Lffwd
Nsteps = Int((p.train_time - p.stim_off)/dt)
ffwdRate = ffwdRate_mean*ones(Nsteps, numFfwd) #funSample(p,utarg)

#----- OU ----#
mu = ffwdRate_mean
bou = 1/400;
sig = 0.2; # std(udrive) ~ 0.04
# sig = sig_list[oo]
for i = 1:Nsteps-1
    ffwdRate[i+1,:] = ffwdRate[i,:] + bou*(mu .- ffwdRate[i,:])*dt + sig*sqrt(dt)*randn(numFfwd);
end

ffwdRate = funMovAvg(ffwdRate, 500)
idx = ffwdRate .< 0
ffwdRate[idx] .= 0.0


# timev = 0.1*collect(1:Nsteps)
# figure()
# for i = 1:4
#     subplot(2,2,i)
#     plot(timev, ffwdRate[:,i], color="black")
#     xlabel("time (ms)")
#     ylabel("spk/s")
#     # ylim([0, 15])
# end
# tight_layout()

# savefig("udrive$(mm).png", dpi=300)


return ffwdRate



end