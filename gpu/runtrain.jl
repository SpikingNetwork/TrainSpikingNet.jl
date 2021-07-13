function runtrain(p,P,Px,w0Index,w0Weights,nc0,stim,xtarg,wpIndexIn,wpIndexOut,wpIndexConvert,wpWeightIn,wpWeightOut,ncpIn,ncpOut)

CUDA.allowscalar(false)

# copy simulation param
nloop = copy(p.nloop)                       # number of training iterations
penlambda = copy(p.penlambda)               # L2-penalty
penlamEE = copy(p.penlamEE)                 # not used
penlamIE = copy(p.penlamIE)                 # not used
penlamEI = copy(p.penlamEI)                 # not used
penlamII = copy(p.penlamII)                 # not used
penmu = copy(p.penmu)                       # Rowsum-penalty
frac = copy(p.frac)                         # not used
learn_every = copy(p.learn_every)           # recursive least squares algorithm updates the plastic weights every learn_every (=10ms)
stim_on = copy(p.stim_on)                   # time at which the stimulus triggering the learned response is turned on (800ms) 
stim_off = copy(p.stim_off)                 # time at which the stimulus triggering the learned response is turned off (1000ms)
train_time = copy(p.train_time)             # total training time (2000ms)
dt = copy(p.dt)                             # simulation time step (0.1ms)
Nsteps = copy(p.Nsteps)                     # number of simulation time steps
Ncells = copy(p.Ncells)                     # number of cells
Ne = copy(p.Ne)                             # number of excitatory cells
Ni = copy(p.Ni)                             # number of inhibitory cells
taue = copy(p.taue)                         # membrane time constant of excitatory cells (10ms)
taui = copy(p.taui)                         # membrane time constant of inhibitory cells (10ms)
sqrtK = copy(p.sqrtK)                       # sqrt(K) where K is the average number of exc/inh synaptic connections to a neuron
threshe = copy(p.threshe)                   # spike threshold of excitatory cells (1)
threshi = copy(p.threshi)                   # spike threshold of inhibitory cells (1)
refrac = copy(p.refrac)                     # refractory period (0, no refractory period)
vre = copy(p.vre)                           # voltage reset after spike (0)
muemin = copy(p.muemin)                     # external input to excitatory neurons (min)
muemax = copy(p.muemax)                     # external input to excitatory neurons (max)
muimin = copy(p.muimin)                     # external input to inhibitory neurons (min)
muimax = copy(p.muimax)                     # external input to inhibitory neurons (max)
tauedecay = copy(p.tauedecay)               # excitatory synaptic decay time constant (3ms) - for balanced connectivity (static)
tauidecay = copy(p.tauidecay)               # inhibitory synaptic decay time constant (3ms) - for balanced connectivity (static)
taudecay_plastic = copy(p.taudecay_plastic) # synaptic decay time constant (150ms) - for plastic connectivity 
maxrate = copy(p.maxrate)                   # maximum firing rate allowed (500Hz)

# set up variables
mu = zeros(Ncells)
mu[1:Ne] = (muemax-muemin)*rand(Ne) .+ muemin
mu[(Ne+1):Ncells] = (muimax-muimin)*rand(Ni) .+ muimin

thresh = zeros(Ncells)
thresh[1:Ne] .= threshe
thresh[(1+Ne):Ncells] .= threshi

invtau = zeros(Ncells)
invtau[1:Ne] .= 1/taue
invtau[(1+Ne):Ncells] .= 1/taui

maxTimes = round(Int,maxrate*train_time/1000)
ns = zeros(Int,Ncells)

forwardInputsE = zeros(Ncells+1)              # excitatory synaptic currents to neurons via balanced connections at one time step
forwardInputsI = zeros(Ncells+1)              # inhibitory synaptic currents to neurons via balanced connections at one time step
forwardInputsP = zeros(Ncells+1)              # synaptic currents to neurons via plastic connections at one time step
forwardInputsEPrev = zeros(Ncells+1)          # copy of forwardInputsE from previous time step
forwardInputsIPrev = zeros(Ncells+1)          # copy of forwardInputsI from previous time step
forwardInputsPPrev = zeros(Ncells+1)          # copy of forwardInputsP from previous time step
forwardSpike = zeros(Ncells)                # spikes emitted by each neuron at one time step
forwardSpikePrev = zeros(Ncells)            # copy of forwardSpike from previous time step

xedecay = zeros(Ncells)                     # synapse-filtered excitatory current (i.e. filtered version of forwardInputsE)
xidecay = zeros(Ncells)                     # synapse-filtered inhibitory current (i.e. filtered version of forwardInputsI)
xpdecay = zeros(Ncells)                     # synapse-filtered plastic current (i.e. filtered version of forwardInputsP)
synInputBalanced = zeros(Ncells)            # sum of xedecay and xidecay (i.e. synaptic current from the balanced connections)
r = zeros(Ncells)                           # synapse-filtered spikes (i.e. filtered version of forwardSpike)

bias = zeros(Ncells)                        # total external input to neurons
lastSpike = -100.0*ones(Ncells)             # last time a neuron spiked
bnotrefrac = Vector{Bool}(undef, Ncells)
bspike = Vector{Bool}(undef, Ncells)
t = 0.0                                     # simulation time (ms)

plusone = convert(FloatPrecision, 1.0)
minusone = convert(FloatPrecision, -1.0)

Px = CuArray(Px);
P = CuArray{FloatPrecision}(P);
r = CuArray{FloatPrecision}(r);
wpWeightIn = CuArray{FloatPrecision}(wpWeightIn);
synInputBalanced = CuArray{FloatPrecision}(synInputBalanced);
stim = CuArray{FloatPrecision}(stim);
xtarg = CuArray{FloatPrecision}(xtarg);
xedecay = CuArray{FloatPrecision}(xedecay);
xidecay = CuArray{FloatPrecision}(xidecay);
xpdecay = CuArray{FloatPrecision}(xpdecay);
forwardInputsEPrev = CuArray{FloatPrecision}(forwardInputsEPrev);
forwardInputsIPrev = CuArray{FloatPrecision}(forwardInputsIPrev);
forwardInputsPPrev = CuArray{FloatPrecision}(forwardInputsPPrev);
forwardInputsE = CuArray{FloatPrecision}(forwardInputsE);
forwardInputsI = CuArray{FloatPrecision}(forwardInputsI);
forwardInputsP = CuArray{FloatPrecision}(forwardInputsP);
forwardSpikePrev = CuArray{FloatPrecision}(forwardSpikePrev);
invtauedecay = convert(FloatPrecision, 1/tauedecay)
invtauidecay = convert(FloatPrecision, 1/tauidecay)
invtaudecay_plastic = convert(FloatPrecision, 1/taudecay_plastic)
dt = convert(FloatPrecision, p.dt)
ncpIn = convert(Array{IntPrecision}, ncpIn)
w0Index = CuArray{IntPrecision}(w0Index)
w0Weights = CuArray{FloatPrecision}(w0Weights)
wpIndexIn = CuArray{IntPrecision}(wpIndexIn)
wpIndexConvert = CuArray{IntPrecision}(wpIndexConvert)
wpIndexOut = CuArray{IntPrecision}(wpIndexOut)
wpWeightOut = CuArray{FloatPrecision}(wpWeightOut)
bias = CuArray{FloatPrecision}(bias)
mu = CuArray{FloatPrecision}(mu)
forwardSpike = CuArray{FloatPrecision}(forwardSpike)
lastSpike = CuArray{FloatPrecision}(lastSpike)
bnotrefrac = CuArray(bnotrefrac)
bspike = CuArray(bspike)
ns = CuArray{IntPrecision}(ns)
invtau = CuArray{FloatPrecision}(invtau)
thresh = CuArray{FloatPrecision}(thresh)

function kernelEI(ispike, w0Index, w0Weights, forwardInputsE, forwardInputsI)
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    istride = blockDim().x * gridDim().x
    jstride = blockDim().y * gridDim().y

    @inbounds for i=i0:istride:size(w0Index,1), j=j0:jstride:length(ispike)
        @atomic forwardInputsE[0x1 + w0Index[i,ispike[j]]] += max(w0Weights[i,ispike[j]], 0)
        @atomic forwardInputsI[0x1 + w0Index[i,ispike[j]]] += min(w0Weights[i,ispike[j]], 0)
    end
    return nothing
end

function kernelP(ispike, wpIndexOut, wpWeightOut, forwardInputsP)
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    istride = blockDim().x * gridDim().x
    jstride = blockDim().y * gridDim().y

    @inbounds for i=i0:istride:size(wpIndexOut,1), j=j0:jstride:length(ispike)
        @atomic forwardInputsP[0x1 + wpIndexOut[i,ispike[j]]] += wpWeightOut[i,ispike[j]]
    end
    return nothing
end

function configurator(config, size_weights)
    xthreads = min(32, size_weights[1])
    ythreads = min(cld(config.threads, xthreads), cld(prod(size_weights), xthreads))
    xblocks = cld(size_weights[1], xthreads)
    yblocks = cld(size_weights[2], ythreads)

    return (threads=(xthreads, ythreads), blocks=(xblocks, yblocks))
end

cukernelEI = cufunction(kernelEI, Tuple{CuDeviceArray{UInt64,1,AS.Global}, CuDeviceArray{IntPrecision,2,AS.Global}, CuDeviceArray{FloatPrecision,2,AS.Global}, CuDeviceArray{FloatPrecision,1,AS.Global}, CuDeviceArray{FloatPrecision,1,AS.Global}})

cukernelP = cufunction(kernelP, Tuple{CuDeviceArray{UInt64,1,AS.Global}, CuDeviceArray{IntPrecision,2,AS.Global}, CuDeviceArray{FloatPrecision,2,AS.Global}, CuDeviceArray{FloatPrecision,1,AS.Global}})

k = CuArray{FloatPrecision}(undef, 2*L, 1, Ncells)
den = CuArray{FloatPrecision}(undef, 1, 1, Ncells)
e = CuArray{FloatPrecision}(undef, 1, 1, Ncells)

# start training loops
for iloop =1:nloop
    println("Loop no. ",iloop) 

    lastSpike .= -100.0
    ns .= 0
    xedecay .= 0
    xidecay .= 0
    xpdecay .= 0
    r .= 0
    v = CuArray(rand(FloatPrecision, Ncells))
    learn_seq = 1

    start_time = time()

    for ti=1:Nsteps
        t = dt*ti;

        forwardInputsE .= 0.0;
        forwardInputsI .= 0.0;
        forwardInputsP .= 0.0;

        if t > Int(stim_off) && t <= Int(train_time) && mod(t, learn_every) == 0
            wpWeightIn, wpWeightOut, learn_seq = rls(k, den, e, p, r, Px, P, synInputBalanced, xtarg, learn_seq, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn, wpWeightOut, plusone, minusone)
        end

        xedecay .+= (-dt.*xedecay .+ forwardInputsEPrev[2:end]).*invtauedecay
        xidecay .+= (-dt.*xidecay .+ forwardInputsIPrev[2:end]).*invtauidecay
        xpdecay .+= (-dt.*xpdecay .+ forwardInputsPPrev[2:end]).*invtaudecay_plastic
        synInputBalanced .= xedecay .+ xidecay

        r .+= (-dt.*r .+ forwardSpikePrev).*invtaudecay_plastic

        if t > Int(stim_on) && t < Int(stim_off)
            bias .= mu .+ stim[ti-round(Int,stim_on/dt),:]
        else
            bias .= mu
        end

        bnotrefrac .= t .> (lastSpike .+ refrac)
        v .+= bnotrefrac.*dt.*(invtau.*(bias .- v .+ synInputBalanced .+ xpdecay))

        bspike .= bnotrefrac .& (v .> thresh)
        forwardSpike .= bspike
        ns .+= bspike
        v .= ifelse.(bspike, vre, v)
        lastSpike .= ifelse.(bspike, t, lastSpike)

        ispike = findall(bspike)  ### this single line accounts for a quarter of the time
        if length(ispike)>0
            configEI = configurator(CUDA.launch_configuration(cukernelEI.fun), (size(w0Weights,1),length(ispike)))
            @cuda name="update_forwardInputsEI" threads=configEI.threads blocks=configEI.blocks kernelEI(ispike, w0Index, w0Weights, forwardInputsE, forwardInputsI)

            configP = configurator(CUDA.launch_configuration(cukernelP.fun), (size(wpWeightOut,1),length(ispike)))
            @cuda name="update_forwardInputsP" threads=configP.threads blocks=configP.blocks kernelP(ispike, wpIndexOut, wpWeightOut, forwardInputsP)
        end

        forwardInputsEPrev = copy(forwardInputsE)
        forwardInputsIPrev = copy(forwardInputsI)
        forwardInputsPPrev = copy(forwardInputsP)
        forwardSpikePrev = copy(forwardSpike)
    end

elapsed_time = time()-start_time
println("elapsed time: ",elapsed_time)

end

return wpWeightIn, wpWeightOut

end
