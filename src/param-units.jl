# import the desired units in this code block.  current only Unitful.jl is supported.
init_code = quote
    using Unitful
    import Unitful: V, s, ms, μs, A, Hz

    # https://github.com/PainterQubits/Unitful.jl/issues/644
    import Unitful: ustrip
    @inline ustrip(A::StridedArray{Q}) where {Q <: Quantity} = reinterpret(Unitful.numtype(Q), A)
end
eval(init_code)

# --- simulation --- #
PType=Symmetric  # storage format of the covariance matrix;  use SymmetricPacked for large models
PPrecision = Float32  # precision of the covariance matrix.  can be Float16 or even <:Integer on GPUs
PScale = 1  # if PPrecision<:Integer then PScale should be e.g. 2^(nbits-2)
PCompute = :fast  # or :small
FloatPrecision = Float32  # precision of all other floating point variables, except time
IntPrecision = UInt32  # precision of all integer variables.  should be > Ncells

# promote_type(PPrecision, typeof(PScale), FloatPrecision) is used to
# accumulate intermediate `gemv` etc. values on GPU, so if PPrecision and
# FloatPrecision are small, make typeof(PScale) big and float

example_neurons = 25  # no. of neurons to save for visualization 
wid = 50ms  # width of the moving average window in time
maxrate = 500Hz # maximum average firing rate; spikes will be lost if the average firing rate exceeds this value

# for exact reproducibility set seed to something and use MersenneTwister for rng_func
seed = nothing
rng_func = (; :gpu => :(CUDA.RNG()), :cpu => :(Random.default_rng()))
rng = eval(rng_func.cpu)
isnothing(seed) || Random.seed!(rng, seed)
save(joinpath(@__DIR__, "rng-init.jld2"), "rng", rng)

dt = 100μs  # simulation timestep


# --- network --- #
Ncells = 4096
Ne = floor(Int, Ncells*0.5)
Ni = ceil(Int, Ncells*0.5)


# --- epoch --- #
train_duration = 1000.0ms
stim_on        = 800.0ms
stim_off       = 1000.0ms
train_time     = stim_off + train_duration

Nsteps = round(Int, train_time/dt)
u0_skip_time = 1000ms
u0_ncells = 1000


# --- neuron --- #
refrac = 100μs    # refractory period
vre = 0.0V          # reset voltage
tau_bale = 3ms    # synaptic time constants
tau_bali = 3ms
tau_plas = 150ms  # can be a vector too, e.g. (150-70)*rand(rng, Ncells) .+ 70

#membrane time constants
tau_meme = 10ms
tau_memi = 10ms
invtau_mem = Vector{eltype(1/tau_meme)}(undef, Ncells)
invtau_mem[1:Ne] .= 1 ./ tau_meme
invtau_mem[(1+Ne):Ncells] .= 1 ./ tau_memi

#spike thresholds
threshe = 1.0V
threshi = 1.0V
thresh = Vector{eltype(vre)}(undef, Ncells)
thresh[1:Ne] .= threshe
thresh[(1+Ne):Ncells] .= threshi

cellModel_file = "cellModel-LIF-units.jl"
cellModel_args = (; thresh, invtau_mem, vre, dt)


g = ustrip(upreferred(threshe-vre)) * 1.0A


# --- fixed connections plugin --- #
pree = prie = prei = prii = 0.1
K = round(Int, Ne*pree)
sqrtK = sqrt(K)

je = 2.0 / sqrtK * tau_meme * g
ji = 2.0 / sqrtK * tau_meme * g 
jx = 0.08 * sqrtK * g 

genStaticWeights_file = "genStaticWeights-erdos-renyi.jl"
genStaticWeights_args = (; K, Ncells, Ne, rng, seed,
                           :jee => 0.15je, :jie => je, :jei => -0.75ji, :jii => -ji)


# --- external stimulus plugin --- #
genXStim_file = "genXStim-ornstein-uhlenbeck.jl"
genXStim_args = (; stim_on, stim_off, dt, Ncells, rng, seed,
                   :mu => 0.0*g,
                   :b => 1000/20s,
                   :sig => 0.2*g*sqrt(1000)/sqrt(1s))


# --- learning --- #
penlambda   = 0.8   # 1 / learning rate
penlamFF    = 1.0
penmu       = 0.01   # regularize weights
learn_every = 10.0ms
PHistory    = 100.0ms  # only relevant if PCompute == :small

correlation_var = K>0 ? :utotal : :uplastic

# a function called at each iteration to query which task to use while training
choose_task_func = :((iloop, ntasks) -> iloop % ntasks + 1)   # or e.g. rand(1:ntasks)


# --- target synaptic current plugin --- #
genUTarget_file = "genUTarget-sinusoids.jl"
genUTarget_args = (; train_time, stim_off, learn_every, Ncells, Nsteps, dt, rng, seed,
                     :Amp => 0.5*g, :period => 1.0s, :biasType => :zero,
                     :mu_ou_bias => 0.0*g,
                     :b_ou_bias => 1000/400s,
                     :sig_ou_bias => 0.02*g*sqrt(1000)/sqrt(1s))



# --- learned connections plugin --- #
L = round(Int,sqrt(K)*2.0)  # number of plastic weights per neuron
Lexc = L
Linh = L
LX = 0

wpscale = sqrt(L) * 2.0

genPlasticWeights_file = "genPlasticWeights-erdos-renyi.jl"
genPlasticWeights_args = (; Ncells, Ne, Lexc, Linh, LX, rng, seed,
                            :frac => 1.0,
                            :wpee => 2.0 * tau_meme * g / wpscale,
                            :wpie => 2.0 * tau_meme * g / wpscale,
                            :wpei => -2.0 * tau_meme * g / wpscale,
                            :wpii => -2.0 * tau_meme * g / wpscale,
                            :wpX => 0.0 * tau_meme * g / wpscale)


# --- feed forward neuron plugin --- #
genRateX_file = "genRateX-ornstein-uhlenbeck.jl"
genRateX_args = (; train_time, stim_off, dt, rng, LX,
                   :mu => 5, :bou => 1000/400s, :sig => 0.2*sqrt(1000)/sqrt(1s), :wid => 500)


# --- external input --- #
X_bale = jx*1.5
X_bali = jx

X_bal = Vector{eltype(g)}(undef, Ncells)
X_bal[1:Ne] .= X_bale
X_bal[Ne+1:Ncells] .= X_bali


# --- time-varying noise --- #
noise_model=:voltage  # or :current
sig = 0  # std dev of the Gaussian noise.  can be vector too.  0.65 works


benchmark = false  # set to true to not update the weights thereby maintaining firing rate
