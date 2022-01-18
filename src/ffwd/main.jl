using Distributions
using PyCall
using PyPlot
using DelimitedFiles
using LinearAlgebra
using Random
using SparseArrays
using JLD

# modifiable parameters

# lam_list = [0.04, 0.1, 0.5, 1.0]
# lam_list = [0.5, 1.0]
# jj = 2


trainEI_list = ["exc", "inh"]
ii = parse(Int64,ARGS[1]) 
# ii = 1
trainEI = trainEI_list[ii]


fracTrained_list = collect(0.2:0.2:1.0)
jj = parse(Int64,ARGS[2]) 
# jj = 10


# L_list = [2.0, 4.0, 6.0, 8.0, 10.0]
L_list = [1.0, 2.0, 3.0, 4.0, 5.0]
# L_list = [55, 110, 165, 220, 275]
kk = parse(Int64,ARGS[3]) # L = 2.0
# kk = 5


# Lffwd_list = [100, 200, 300, 400]
Lffwd_list = [1, 100, 200, 300]
ll = parse(Int64,ARGS[4])
# ll = 1


### wpffwd_list = [0.0, 1.0, 2.0]
### pp = 1


include("param.jl")
include("genWeights.jl")
include("genPlasticWeights.jl")
include("convertWgtIn2Out.jl")
include("genffwdRate.jl")
include("genStim.jl")
include("genCellsTrained.jl")
include("runinitial.jl")
include("runtrain.jl")
include("runtest.jl")
include("funMovAvg.jl")
include("funCorrTarg.jl")
include("funCorrDecomp.jl")
include("funSample.jl")
include("funRollingAvg.jl")
include("runperformance.jl")
include("calcWeights.jl")
include("replaceWp.jl")


dirdata = "/data/kimchm/data/dale/janelia/trained/synthetic/" * trainEI * "/frac$(jj)/L$(kk)/Lffwd$(ll)/"
dirtarget_synthetic = "/data/kimchm/data/dale/janelia/s1alm/target/synthetic/"
dirutarg_pyr_lickright = "/data/kimchm/data/dale/janelia/s1alm/target/synthetic/utarg/pyr/lickright/"
dirutarg_pyr_lickleft = "/data/kimchm/data/dale/janelia/s1alm/target/synthetic/utarg/pyr/lickleft/"
dirutarg_fs_lickright = "/data/kimchm/data/dale/janelia/s1alm/target/synthetic/utarg/fs/lickright/"
dirutarg_fs_lickleft = "/data/kimchm/data/dale/janelia/s1alm/target/synthetic/utarg/fs/lickleft/"
dirfig = "/data/kimchm/data/dale/janelia/figure/synthetic/trained/"


if ~ispath(dirdata)
    mkpath(dirdata)
    mkpath(dirfig)
end


#----------- initialization --------------#

# run initial balanced network
w0Index, w0Weights, nc0 = genWeights(p)
uavg, ns0, ustd = runinitial(p,w0Index, w0Weights, nc0)

# select subpopulation to be trained
if trainEI == "exc"                                             ########### TRAIN EXC / INH
    rtarg_lickright = load(dirtarget_synthetic * "synthetic_Pyr_lickright.jld", "Pyr")
    rtarg_lickleft = load(dirtarget_synthetic * "synthetic_Pyr_lickleft.jld", "Pyr")
elseif trainEI == "inh"
    rtarg_lickright = load(dirtarget_synthetic * "synthetic_FS_lickright.jld", "FS")
    rtarg_lickleft = load(dirtarget_synthetic * "synthetic_FS_lickleft.jld", "FS")
end

# sample a subset of neurons to be trained
Ntrained = Int(p.Ne * p.fracTrained)                                 ########### NUMBER OF TRAINED NEURONS
sampledNeurons = sort(shuffle(collect(1:p.Ne))[1:Ntrained])
rtarg_lickright = rtarg_lickright[:, sampledNeurons]
rtarg_lickleft = rtarg_lickleft[:, sampledNeurons]
rtarg_mean = (rtarg_lickright + rtarg_lickleft)/2
almOrd, matchedCells = genCellsTrained(dirfig, p, rtarg_mean, ns0, trainEI)

#####################################################
################ Feedforward inputs #################
#####################################################                    
# select plastic weights to be trained
wpWeightFfwd, wpWeightIn, wpWeightOut, wpIndexIn, wpIndexOut, wpIndexConvert, ncpIn, ncpOut = genPlasticWeights(p, w0Index, nc0, ns0, matchedCells, trainEI)

stim = Vector{Array{Float64,2}}()
stim_R = genStim(p)
stim_L = genStim(p)
push!(stim, stim_R)
push!(stim, stim_L)

# load targets
xtarg = Vector{Array{Float64,2}}()
if trainEI == "exc"
    utarg_R = load(dirtarget_synthetic * "utarg_pyr_lickright.jld", "utarg")[:,sampledNeurons]
    utarg_L = load(dirtarget_synthetic * "utarg_pyr_lickleft.jld", "utarg")[:,sampledNeurons]
elseif trainEI == "inh"
    utarg_R = load(dirtarget_synthetic * "utarg_fs_lickright.jld", "utarg")[:,sampledNeurons]
    utarg_L = load(dirtarget_synthetic * "utarg_fs_lickleft.jld", "utarg")[:,sampledNeurons]
end
push!(xtarg, utarg_R)
push!(xtarg, utarg_L)

#####################################################
################ Feedforward inputs #################
#####################################################                    
# generate ffwd input
ffwdRate_mean = 5.0
ffwdRate = Vector{Array{Float64,2}}()
ffwdRate_R = genffwdRate(p, ffwdRate_mean)
ffwdRate_L = genffwdRate(p, ffwdRate_mean)
push!(ffwdRate, ffwdRate_R)
push!(ffwdRate, ffwdRate_L)

#----------- save files --------------#
fname_param = dirdata * "p.jld"
fname_w0Index = dirdata * "w0Index.jld"
fname_w0Weights = dirdata * "w0Weights.jld"
fname_nc0 = dirdata * "nc0.jld"
fname_wpIndexIn = dirdata * "wpIndexIn.jld"
fname_wpIndexOut = dirdata * "wpIndexOut.jld"
fname_wpIndexConvert = dirdata * "wpIndexConvert.jld"
fname_ncpIn = dirdata * "ncpIn.jld"
fname_ncpOut = dirdata * "ncpOut.jld"
fname_stim_R = dirdata * "stim_R.jld"
fname_stim_L = dirdata * "stim_L.jld"
fname_ffwdRate = dirdata * "ffwdRate.jld"
# fname_ffwdRate_L = dirdata * "ffwdRate_L.jld"

fname_uavg = dirdata * "uavg.jld"
fname_almOrd = dirdata * "almOrd.jld"
fname_matchedCells = dirdata * "matchedCells.jld"

save(fname_param,"p", p)
save(fname_w0Index,"w0Index", w0Index)
save(fname_w0Weights,"w0Weights", w0Weights)
save(fname_nc0,"nc0", nc0)
save(fname_wpIndexIn,"wpIndexIn", wpIndexIn)
save(fname_wpIndexOut,"wpIndexOut", wpIndexOut)
save(fname_wpIndexConvert,"wpIndexConvert", wpIndexConvert)
save(fname_ncpIn,"ncpIn", ncpIn)
save(fname_ncpOut,"ncpOut", ncpOut)
save(fname_stim_R,"stim", stim_R)
save(fname_stim_L,"stim", stim_L)
save(fname_ffwdRate,"ffwdRate", ffwdRate)
# save(fname_ffwdRate_L,"ffwdRate", ffwdRate_L)

save(fname_uavg,"uavg", uavg)
save(fname_almOrd,"almOrd", almOrd)
save(fname_matchedCells,"matchedCells", matchedCells)

#####################################################
################ Feedforward inputs #################
#####################################################                    
#----------- run train --------------#
wpWeightIn, wpWeightOut, wpWeightFfwd = runtrain(dirdata,p,w0Index,w0Weights,nc0, stim, xtarg,
wpWeightFfwd, wpIndexIn, wpIndexOut, wpIndexConvert, wpWeightIn, wpWeightOut, ncpIn, ncpOut, 
almOrd, matchedCells, ffwdRate)


#----------- save files --------------#
fname_wpWeightIn = dirdata * "wpWeightIn.jld"
fname_wpWeightOut = dirdata * "wpWeightOut.jld"
fname_wpWeightFfwd = dirdata * "wpWeightFfwd.jld"
# fname_performance_test = dirdata * "_performance_test.jld"
# fname_frac_switch_exc = dirdata * "_frac_switch_exc.jld"
# fname_frac_switch_inh = dirdata * "_frac_switch_inh.jld"

save(fname_wpWeightIn,"wpWeightIn", wpWeightIn)
save(fname_wpWeightOut,"wpWeightOut", wpWeightOut)
save(fname_wpWeightFfwd,"wpWeightFfwd", wpWeightFfwd)
# save(fname_performance_test,"performance_test", performance_test)
# save(fname_frac_switch_exc,"frac_switch_exc", frac_switch_exc)
# save(fname_frac_switch_inh,"frac_switch_inh", frac_switch_inh)
