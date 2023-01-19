# --- simulation --- #
PType=Symmetric  # storage format of the covariance matrix;  use SymmetricPacked for large models
PPrecision = Float64  # precision of the covariance matrix.  can be <:Integer on GPUs
PScale = 1  # if PPrecision<:Integer then PScale should be e.g. 2^(nbits-2)
FloatPrecision = Float64  # precision of all other floating point variables, except time
IntPrecision = UInt32  # precision of all integer variables

example_neurons = 25  # no. of neurons to save for visualization 
wid = 50  # width (ms) of the moving average window in time
maxrate = 500 # (Hz) maximum average firing rate; spikes will be lost if the average firing rate exceeds this value

seed = nothing
rng_func = Dict("gpu"=>:(CUDA.RNG()), "cpu"=>:(Random.default_rng()))
rng = eval(rng_func["cpu"])
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
genStim_file = "genStim-ornstein-uhlenbeck.jl"
genStim_args = Dict(:stim_on => stim_on, :stim_off => stim_off, :dt => dt, :Ncells => Ncells,
                    :mu => 0.0, :b => 1/20, :sig => 0.2,
                    :rng => rng)


# --- neuron --- #
taue = 10      # membrane time constant (ms)
taui = 10 
threshe = 1.0  # spike threshold
threshi = 1.0   
refrac = 0.1   # refractory period
vre = 0.0      # reset voltage
tauedecay = 3  # synaptic time constants (ms) 
tauidecay = 3
taudecay_plastic = 150  # can be a vector too, e.g. (150-70)*rand(rng, Ncells) .+ 70


# --- fixed connections plugin --- #
pree = prie = prei = prii = 0.1
K = round(Int, Ne*pree)
sqrtK = sqrt(K)

genStaticWeights_file = "genStaticWeights-erdos-renyi.jl"
genStaticWeights_args = Dict(:K => K, :Ncells => Ncells, :Ne => Ne,
                             :pree => pree, :prie => prie, :prei => prei, :prii => prii,
                             :rng => rng)

g = 1.0
je = 2.0 / sqrtK * taue * g
ji = 2.0 / sqrtK * taue * g 
jx = 0.08 * sqrtK * g 

merge!(genStaticWeights_args, Dict(:jee => 0.15je, :jie => je, :jei => -0.75ji, :jii => -ji))


# --- learning --- #
penlambda   = 0.8   # 1 / learning rate
penlamFF    = 1.0
penmu       = 0.01  # regularize weights
learn_every = 10.0  # (ms)

correlation_var = K>0 ? :xtotal : :xplastic

choose_task_func = :((iloop, ntasks) -> iloop % ntasks + 1)   # or e.g. rand(1:ntasks)


# --- target synaptic current plugin --- #
genTarget_file = "genTarget-sinusoids.jl"
genTarget_args = Dict(:train_time => train_time, :stim_off => stim_off, :learn_every => learn_every,
                      :Ncells => Ncells, :Nsteps => Nsteps, :dt => dt,
                      :A => 0.5, :period => 1000.0, :biasType => :zero,
                      :mu_ou_bias => 0.0, :b_ou_bias => 1/400, :sig_ou_bias => 0.02,
                      :rng => rng)

# --- learned connections plugin --- #
L = round(Int,sqrt(K)*2.0)  # number of plastic weights per neuron
Lexc = L
Linh = L
Lffwd = 0

wpscale = sqrt(L) * 2.0

genPlasticWeights_file = "genPlasticWeights-erdos-renyi.jl"
genPlasticWeights_args = Dict(:Ncells => Ncells, :frac => 1.0, :Ne => Ne,
                              :L => L, :Lexc => Lexc, :Linh => Linh, :Lffwd => Lffwd,
                              :wpee => 2.0 * taue * g / wpscale,
                              :wpie => 2.0 * taue * g / wpscale,
                              :wpei => -2.0 * taue * g / wpscale,
                              :wpii => -2.0 * taue * g / wpscale,
                              :wpffwd => 0,
                              :rng => rng)


# --- feed forward neuron plugin --- #
genFfwdRate_file = "genFfwdRate-ornstein-uhlenbeck.jl"
genFfwdRate_args = Dict(:train_time => train_time, :stim_off => stim_off, :dt => dt,
                        :Lffwd => Lffwd, :mu => 5, :bou => 1/400, :sig => 0.2, :wid => 500,
                        :rng => rng)


# --- external input --- #
muemin = jx*1.5
muemax = jx*1.5
muimin = jx
muimax = jx

mu = Vector{Float64}(undef, Ncells)
mu[1:Ne] = (muemax-muemin) * rand(rng, Ne) .+ muemin
mu[(Ne+1):Ncells] = (muimax-muimin) * rand(rng, Ni) .+ muimin


# --- time-varying noise --- #
noise_model=:current  # or :voltage
sig = 0  # std dev of the Gaussian noise
