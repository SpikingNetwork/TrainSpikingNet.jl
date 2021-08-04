FloatPrecision = Float64
IntPrecision = UInt64
seed=1

performance_interval = 0  # set to N to measure every N loops
example_neurons = 25
wid = 50

dt = 0.1 #simulation timestep (ms)

# training variables
nloop          = 1; #200
penlambda      = 3.0; # 0.1 or 0.5 
penlamEE       = 3.0; # 2.0
penlamEI       = 3.0; # 0.08
penlamIE       = 3.0; # 2.0
penlamII       = 3.0; # 0.08
penmu          = 8.0; # 2.0
frac           = 1.0;
learn_every    = 10.0 # (ms)

# innate, train, test time (ms)
train_duration = 1000.;
stim_on        = 800.;
stim_off       = 1000.;
train_time     = stim_off + train_duration;

Nsteps = round(Int, train_time/dt)

# neuron param      
taue = 10; #membrane time constant for exc. neurons (ms)
taui = 10; 
threshe = 1.0 # spike threshold
threshi = 1.0   
refrac = 0.1 # refractory period
vre = 0.0

#synaptic time constants (ms) 
tauedecay = 3
tauidecay = 3
taudecay_plastic = 150 

# network size      
Ncells = 2000;
Ne = round(Int, Ncells*0.5);
Ni = round(Int, Ncells*0.5);

if Ncells == typemax(IntPrecision)
  @warn "IntPrecision is too small for GPU (but fine for CPU)"
elseif Ncells > typemax(IntPrecision)
  @error "IntPrecision is too small"
end

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

# plastic weights
L = round(Int,sqrt(K)*2.0) # number of exc/inh plastic weights per neuron
Lexc = L # excitatory L
Linh = L # inhibitory L
wpscale = sqrt(L) * 2.0

wpee = je * sqrtK / wpscale
wpie = je * sqrtK / wpscale
wpei = -ji * sqrtK / wpscale
wpii = -ji * sqrtK / wpscale


maxrate = 500 #(Hz) maximum average firing rate.  if the average firing rate across the simulation for any neuron exceeds this value, some of that neuron's spikes will not be saved


p = paramType(FloatPrecision,IntPrecision,seed,performance_interval,example_neurons,wid,train_duration,nloop,penlambda,penlamEE,penlamEI,penlamIE,penlamII,penmu,frac,learn_every,stim_on,stim_off,train_time,dt,Nsteps,Ncells,Ne,Ni,pree,prei,prie,prii,taue,taui,K,sqrtK,L,Lexc,Linh,wpscale,
je,ji,jx,jee,jei,jie,jii,wpee,wpei,wpie,wpii,muemin,muemax,muimin,muimax,vre,threshe,threshi,refrac,tauedecay,tauidecay,taudecay_plastic,maxrate);
