using Distributions
using PyCall
using PyPlot
using LinearAlgebra
using Random
using JLD
using NLsolve

include("ricciardi.jl")
include("solveRicci.jl")
include("param.jl")
include("genWeights.jl")
include("runinitial2.jl")


dirtarget_selectivity = "/data/kimchm/data/dale/janelia/s1alm/target/selectivity/"
dirutarg_pyr_lickright = "/data/kimchm/data/dale/janelia/s1alm/target/selectivity/utarg/pyr/lickright/"
dirutarg_pyr_lickleft = "/data/kimchm/data/dale/janelia/s1alm/target/selectivity/utarg/pyr/lickleft/"

if ~ispath(dirutarg_pyr_lickright)
    mkpath(dirutarg_pyr_lickright)
    mkpath(dirutarg_pyr_lickleft)
end

# number of Pyr cells = 2602
nid = parse(Int64,ARGS[1])
# nid = 1

#---------- run initial balanced network ----------#
#################################################################################
##### p.train_time = 20000 (20sec) to get nice log-normal rate distribution #####
#################################################################################
w0Index, w0Weights, nc0 = genWeights(p);
times, ns, ustd, uavg = runinitial2(p,w0Index, w0Weights, nc0);
ustd_exc = mean(ustd[1:1000])

#############################
##### lick right / left #####
#############################
for ii = 1:2
    if ii == 1
        fname_rtarg = "movingrate_Pyr_lickleft.jld"
        dirutarg = dirutarg_pyr_lickleft
    elseif ii == 2
        fname_rtarg = "movingrate_Pyr_lickright.jld"
        dirutarg = dirutarg_pyr_lickright
    end

    #---------- mean rate to be learned ----------#
    targetRate = load(dirtarget_selectivity * fname_rtarg, "Pyr")
    idxZero = targetRate .== 0.0
    targetRate[idxZero] .= 0.1

    #---------- initial condition to Ricciardi ----------#
    Ntime = size(targetRate)[2]
    initial_mu = 0.5*ones(Ntime)
    sigma = ustd_exc / sqrt(p.tauedecay * 1.3) # factor 1.3 was calibrated manually
    tau = p.taue/1000.0 
    VT = p.threshe
    Vr = p.vre

    #---------- Solve Ricciardi ----------#
    targetRate_slice = targetRate[nid,:]
    utarg = solveRicci(targetRate_slice, initial_mu, sigma, tau, VT, Vr)

    #---------- Save ----------#
    save(dirutarg * "utarg$(nid).jld", "utarg", utarg)
end


