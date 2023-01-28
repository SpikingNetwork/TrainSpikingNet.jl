# --- simulation --- #
PType=Symmetric  # storage format of the covariance matrix;  use SymmetricPacked for large models
PPrecision = Float32  # precision of the covariance matrix.  can be Float16 or even <:Integer on GPUs
PScale = 1  # if PPrecision<:Integer then PScale should be e.g. 2^(nbits-2)
FloatPrecision = Float32  # precision of all other floating point variables, except time
IntPrecision = UInt32  # precision of all integer variables.  should be > Ncells

# promote_type(PPrecision, typeof(PScale), FloatPrecision) is used to
# accumulate intermediate `gemv` etc. values on GPU, so if PPrecision and
# FloatPrecision are small, make typeof(PScale) big and float

example_neurons = 25  # no. of neurons to save for visualization 
wid = 50  # width (ms) of the moving average window in time
maxrate = 500 # (Hz) maximum average firing rate; spikes will be lost if the average firing rate exceeds this value

seed = nothing
rng_func = (; :gpu => :(CUDA.RNG()), :cpu => :(Random.default_rng()))
rng = eval(rng_func.cpu)
isnothing(seed) || Random.seed!(rng, seed)
save(joinpath(data_dir,"rng-init.jld2"), "rng", rng)

dt = 0.1  # simulation timestep (ms)


# --- network --- #
Ncells = 4096
Ne = floor(Int, Ncells*0.5)
Ni = ceil(Int, Ncells*0.5)


# --- epoch --- #
train_duration = 1000.0  # (ms)
stim_on        = 800.0
stim_off       = 1000.0
train_time     = stim_off + train_duration

Nsteps = round(Int, train_time/dt)


# --- external stimulus plugin --- #
genXStim_file = "genXStim-ornstein-uhlenbeck.jl"
genXStim_args = (; stim_on, stim_off, dt, Ncells, rng,
                   :mu => 0.0, :b => 1/20, :sig => 0.2)


# --- neuron --- #
refrac = 0.1    # refractory period
vre = 0.0       # reset voltage
tau_bale = 3    # synaptic time constants (ms) 
tau_bali = 3
tau_plas = 150  # can be a vector too, e.g. (150-70)*rand(rng, Ncells) .+ 70

#membrane time constants
tau_meme = 10   # (ms)
tau_memi = 10 
invtau_mem = Vector{Float64}(undef, Ncells)
invtau_mem[1:Ne] .= 1 ./ tau_meme
invtau_mem[(1+Ne):Ncells] .= 1 ./ tau_memi

#spike thresholds
threshe = 1.0
threshi = 1.0   
thresh = Vector{Float64}(undef, Ncells)
thresh[1:Ne] .= threshe
thresh[(1+Ne):Ncells] .= threshi

cellModel_file = "cellModel-LIF.jl"
cellModel_args = (; thresh, invtau_mem, vre, dt)


# --- fixed connections plugin --- #
pree = prie = prei = prii = 0.1
K = round(Int, Ne*pree)
sqrtK = sqrt(K)

g = 1.0
je = 2.0 / sqrtK * tau_meme * g
ji = 2.0 / sqrtK * tau_meme * g 
jx = 0.08 * sqrtK * g 

genStaticWeights_file = "genStaticWeights-erdos-renyi.jl"
genStaticWeights_args = (; K, Ncells, Ne, pree, prie, prei, prii, rng,
                           :jee => 0.15je, :jie => je, :jei => -0.75ji, :jii => -ji)


# --- learning --- #
penlambda   = 0.8   # 1 / learning rate
penlamFF    = 1.0
penmu       = 0.01  # regularize weights
learn_every = 10.0  # (ms)

correlation_var = K>0 ? :utotal : :uplastic

choose_task_func = :((iloop, ntasks) -> iloop % ntasks + 1)   # or e.g. rand(1:ntasks)


# --- target synaptic current plugin --- #
genUTarget_file = "genUTarget-sinusoids.jl"
genUTarget_args = (; train_time, stim_off, learn_every, Ncells, Nsteps, dt, rng,
                     :A => 0.5, :period => 1000.0, :biasType => :zero,
                     :mu_ou_bias => 0.0, :b_ou_bias => 1/400, :sig_ou_bias => 0.02)

# --- learned connections plugin --- #
L = round(Int,sqrt(K)*2.0)  # number of plastic weights per neuron
Lexc = L
Linh = L
LX = 0

wpscale = sqrt(L) * 2.0

genPlasticWeights_file = "genPlasticWeights-erdos-renyi.jl"
genPlasticWeights_args = (; Ncells, Ne, L, Lexc, Linh, LX, rng,
                            :frac => 1.0,
                            :wpee => 2.0 * tau_meme * g / wpscale,
                            :wpie => 2.0 * tau_meme * g / wpscale,
                            :wpei => -2.0 * tau_meme * g / wpscale,
                            :wpii => -2.0 * tau_meme * g / wpscale,
                            :wpX => 0)


# --- feed forward neuron plugin --- #
genRateX_file = "genRateX-ornstein-uhlenbeck.jl"
genRateX_args = (; train_time, stim_off, dt, rng, LX,
                   :mu => 5, :bou => 1/400, :sig => 0.2, :wid => 500)


# --- external input --- #
X_bale = jx*1.5
X_bali = jx

X_bal = Vector{Float64}(undef, Ncells)
X_bal[1:Ne] .= X_bale
X_bal[(Ne+1):Ncells] .= X_bali


# --- time-varying noise --- #
noise_model=:current  # or :voltage
sig = 0  # std dev of the Gaussian noise.  can be vector too
