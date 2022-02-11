FloatPrecision = Float64
IntPrecision = UInt32
PType=Symmetric  # or SymmetricPacked for large models
seed=nothing

example_neurons = 25  # no. of neurons to save for visualization 
wid = 50  # width (ms) of the moving average window in time

monitor_resources_used = 0  # set to N to measure every N seconds

rng_func = Dict("gpu"=>:(CUDA.RNG()), "cpu"=>:(Random.default_rng()))
rng = eval(rng_func["cpu"])
isnothing(seed) || Random.seed!(rng, seed)
save(joinpath(parsed_args["data_dir"],"rng-init.jld2"), "rng", rng)

dt = 0.1 #simulation timestep (ms)

# training variables
penlambda      = 0.8  # 1 / learning rate
penmu          = 0.01 # regularize weights
frac           = 1.0
learn_every    = 10.0 # (ms)

# innate, train, test time (ms)
train_duration = 1000.0
stim_on        = 800.0
stim_off       = 1000.0
train_time     = stim_off + train_duration

Nsteps = round(Int, train_time/dt)

# network size
Ncells = 4096
Ne = floor(Int, Ncells*0.5)
Ni = ceil(Int, Ncells*0.5)

if Ncells == typemax(IntPrecision)
  @warn "IntPrecision is too small for GPU (but fine for CPU)"
elseif Ncells > typemax(IntPrecision)
  @error "IntPrecision is too small"
end

# neuron param      
taue = 10 #membrane time constant for exc. neurons (ms)
taui = 10 
threshe = 1.0 # spike threshold
threshi = 1.0   
refrac = 0.1 # refractory period
vre = 0.0

#synaptic time constants (ms) 
tauedecay = 3
tauidecay = 3
taudecay_plastic = 150  # can be a vector too, e.g. (150-70)*rand(rng, Ncells) .+ 70

# connectivity 
pree = 0.1
prei = 0.1
prie = 0.1
prii = 0.1
K = round(Int, Ne*pree)
sqrtK = sqrt(K)

# synaptic strength
g = 1.0 # 1.0, 1.5
je = 2.0 / sqrtK * taue * g
ji = 2.0 / sqrtK * taue * g 
jx = 0.08 * sqrtK * g 

jee = je*0.15
jie = je
jei = -ji*0.75
jii = -ji

muemin = jx*1.5 # exc external input
muemax = jx*1.5
muimin = jx # inh external input
muimax = jx

mu = Vector{Float64}(undef, Ncells)
mu[1:Ne] = (muemax-muemin)*rand(rng, Ne) .+ muemin
mu[(Ne+1):Ncells] = (muimax-muimin)*rand(rng, Ni) .+ muimin

# plastic weights
L = round(Int,sqrt(K)*2.0) # number of exc/inh plastic weights per neuron
Lexc = L # excitatory L
Linh = L # inhibitory L
wpscale = sqrt(L) * 2.0

wpee = 2.0 * taue * g / wpscale
wpie = 2.0 * taue * g / wpscale
wpei = -2.0 * taue * g / wpscale
wpii = -2.0 * taue * g / wpscale

sig0 = 0.65*sqrt(dt)

maxrate = 500 #(Hz) maximum average firing rate.  if the average firing rate across the simulation for any neuron exceeds this value, some of that neuron's spikes will not be saved


p = paramType(FloatPrecision,IntPrecision,PType,seed,rng_func,example_neurons,wid,train_duration,penlambda,penmu,frac,learn_every,stim_on,stim_off,train_time,dt,Nsteps,Ncells,Ne,Ni,pree,prei,prie,prii,taue,taui,K,sqrtK,L,Lexc,Linh,wpscale,
je,ji,jx,jee,jei,jie,jii,wpee,wpei,wpie,wpii,mu,vre,threshe,threshi,refrac,tauedecay,tauidecay,taudecay_plastic,sig0,maxrate)
