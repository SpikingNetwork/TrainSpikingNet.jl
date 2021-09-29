mutable struct paramType
    FloatPrecision::DataType
    IntPrecision::DataType
    seed::Union{Nothing,Int}
    rng::AbstractRNG
    example_neurons::Int
    wid::Int
    train_duration::Float64
    penlambda::Float64
    penlamEE::Float64
    penlamEI::Float64
    penlamIE::Float64
    penlamII::Float64
    penmu::Float64
    frac::Float64
    learn_every::Float64
    stim_on::Float64
    stim_off::Float64
    train_time::Float64
    dt::Float64
    Nsteps::Int64
    Ncells::Int64
    Ne::Int64
    Ni::Int64
    pree::Float64
    prei::Float64
    prie::Float64
    prii::Float64
    taue::Float64
    taui::Float64
    K::Int64
    sqrtK::Float64
    L::Int64
    Lexc::Int64
    Linh::Int64
    wpscale::Float64
    je::Float64
    ji::Float64
    jx::Float64
    jee::Float64
    jei::Float64
    jie::Float64
    jii::Float64
    wpee::Float64
    wpei::Float64
    wpie::Float64
    wpii::Float64
    mu::Vector{Float64}
    vre::Float64
    threshe::Float64
    threshi::Float64
    refrac::Float64
    tauedecay::Float64
    tauidecay::Float64
    taudecay_plastic::Float64    
    sig0::Float64
    maxrate::Float64
end
