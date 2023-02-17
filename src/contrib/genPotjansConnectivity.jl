using SparseArrays
using JLD2
using UnicodePlots
using ProgressMeter
using Distributions, Random
#global verbose
#verbose = false

"""
This file consists of a function stack that seemed necessary to achieve a network with Potjans like wiring in Julia using TrainSpikeNet.jl to simulate.
This code draws heavily on the PyNN OSB Potjans implementation code found here:
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
"""

function potjans_params(ccu, scale=1.0::Float64)
    """
    Hard coded stuff.
    Outputs adapted Potjans parameters.
    """

    # Probabilities for >=1 connection between neurons in the given populations. 
    # The first index is for the target population; the second for the source population
    #             2/3e      2/3i    4e      4i      5e      5i      6e      6i

    conn_probs = [[0.1009,  0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.    ],
                [0.1346,   0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.    ],
                [0.0077,   0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.    ],
                [0.0691,   0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.    ],
                [0.1004,   0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
                [0.0548,   0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.    ],
                [0.0156,   0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
                [0.0364,   0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443]]

    layer_names = ["23E","23I","4E","4I","5E", "5I", "6E", "6I"]
    transform_matrix_ind = zip(collect(1:8),[1,3,5,7,2,4,6,8])
    
    # hard coded stuff is manipulated below:
    columns_conn_probs = [col for col in eachcol(conn_probs)][1]

    conn_probs_= copy(conn_probs)
    
    ## this list gets reorganised to reflect top excitatory bottom inhibitory.
    ## Rearrange the whole matrix so that excitatory connections form a top partition 
    #and inhibitory neurons form a bottom partition.

    cumulative = Dict() 
    v_old=1
    for (k,v) in pairs(ccu)
        ## A cummulative cell count
        cumulative[k]=collect(v_old:v+v_old)
        v_old=v+v_old
    end    
    return (cumulative,ccu,layer_names,columns_conn_probs,conn_probs_)
end

function build_matrix(cumulative::Dict{Any, Any}, conn_probs::Vector{Vector{Float64}},Ncells,weights)
    """
    Iteration logic seperated from synapse selection logic for readability only.
    """
    #Ncells +=1

    edge_dict = Dict() 
    for src in 1:Ncells
        edge_dict[src] = Int64[]
    end    
    w0Index = spzeros(Int,Ncells,Ncells)
    w0Weights = spzeros(Float32,Ncells,Ncells)
    Ne = 0 
    Ni = 0
    Lexc = spzeros(Float32,Ncells,Ncells)
    Linh = spzeros(Float32,Ncells,Ncells)
    @showprogress for (i,(k,v)) in enumerate(pairs(cumulative))
        for src in v
            for (j,(k1,v1)) in enumerate(pairs(cumulative))
                for tgt in v1
                    #src+=1
                    #tgt+=1
                    if src!=tgt
                        prob = conn_probs[i][j]
                        if rand()<prob
                            item = src,tgt,k,k1
                            append!(edge_dict[src],tgt)
                            index_assignment!(item,w0Weights,Ne,Ni,Lexc,Linh,weights)
    
                        end
                    end
                end
            end
        end
    end
    return w0Weights,edge_dict,Ne,Ni,Lexc,Linh
end
function index_assignment!(item,w0Weights,Ne,Ni,Lexc,Linh,weights)  
    """
    Use a nested iterator ideally this will flatten the readability of subsequent code.


    """
    (jee,jie,jei,jii) = weights

    # Mean synaptic weight for all excitatory projections except L4e->L2/3e
    w_mean = 87.8e-3  # nA
    w_ext = 87.8e-3  # nA
    # Mean synaptic weight for L4e->L2/3e connections 
    # See p. 801 of the paper, second paragraph under 'Model Parameterization', 
    # and the caption to Supplementary Fig. 7
    w_234 = 2 * w_mean  # nA
    # Standard deviation of weight distribution relative to mean for L4e->L2/3e
    # This value is not mentioned in the paper, but is chosen to match the 
    # original code by Tobias Potjans
    w_rel_234 = 0.05
    # Relative inhibitory synaptic weight
    wig = -4.
    (src,tgt,k,k1) = item
    if occursin("E",k) 
        if occursin("E",k1)   
            if occursin("4E",k) & occursin("23E",k)
                d = Normal(w_234,w_rel_234)
                td = truncated(d, 0.0, Inf)
                weight = abs(rand(td, 1))
                w0Weights[tgt,src] = weight 
            else         
                w0Weights[tgt,src] = jee
            end
        else# meaning if the same as a logic: occursin("I",k1) is true                   
            w0Weights[tgt,src] = jei
        end
        Lexc[tgt,src] = copy(w0Weights[tgt,src])
        Ne+=1	
    else
        @assert occursin("I",k) 
        # meaning meaning if the same as a logic: elseif occursin("I",k) is true  
        if occursin("E",k1)    
            w0Weights[tgt,src] = wig# -jie 
        else# eaning meaning if the same as a logic: if occursin("I",k1)      is true               
            w0Weights[tgt,src] = wig# -jii  
            @assert occursin("I",k1) 
        end
        Linh[tgt,src] = copy(w0Weights[tgt,src])
        Ni+=1
    end
    # if verbose
    if false
        if Lexc[tgt,src] <0.0
            @show(k,k1,Lexc[tgt,src])
        end
        if Lexc[tgt,src] <0.0
            @show(k,k1,Lexc[tgt,src])
        end
    end        

end

function potjans_weights(Ncells::Int64, jee::Float64, jie::Float64, jei::Float64, jii::Float64, ccu, scale=1.0::Float64)
    (cumulative,ccu,layer_names,_,conn_probs) = potjans_params(ccu,scale)    
    weights = [jee,jie,jei,jii]
    #w_mean = 87.8e-3  # nA
    ###
    # Lower memory footprint motivations.
    # a sparse matrix can be stored as a smaller dense matrix.
    # A 2D matrix should be stored as 1D matrix of srcs,tgts
    # A 2D weight matrix should be stored as 1 matrix, which is redistributed in loops using 
    # the 1D matrix of srcs,tgts.
    ###
    
    w0Weights,edge_dict,Ne,Ni,Lexc,Linh = build_matrix(cumulative,conn_probs,Ncells,weights)
    if verbose 
        @show(maximum(Linh.nzval))
        @show(maximum(Lexc.nzval))
        @show(minimum(Linh.nzval))
        @show(minimum(Lexc.nzval))
    end
    (edge_dict,w0Weights,Ne,Ni,Lexc,Linh)
end

function build_w0Index(edge_dict,Ncells)
    nc0Max = 0
    # what is the maximum out degree of this ragged array?
    for (k,v) in pairs(edge_dict)
        templength = length(v)
        if templength>nc0Max
            nc0Max=templength
        end
    end

    # outdegree
    nc0 = Int.(nc0Max*ones(Ncells))

    ##
    # Force ragged array into smallest dense rectangle (contains zeros for undefined synapses) 
    ##
    w0Index = spzeros(Int64,nc0Max,Ncells)
    for pre_cell = 1:Ncells

        post_cells = edge_dict[pre_cell]
        w0Index[1:length(edge_dict[pre_cell]),pre_cell] = post_cells
    end
    nc0,w0Index
    
end


##
# The following code just gives me the cell count (NCell) upfront, which I need to size arrays in the methods.
##
function get_Ncell(scale=1.0::Float64)
	ccu = Dict("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)
	ccu = Dict((k,ceil(Int64,v*scale)) for (k,v) in pairs(ccu))
	Ncells = sum([i for i in values(ccu)])+1
	Ne = sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]])
    Ncells, Ne, ccu

end


#=
function build_if_nec()
    if !isfile("potjans_matrix.jld2")
        scale =1.0/30.0
        Ncells,Ne, ccu = get_Ncell(scale)
        
        pree = 0.1
        K = round(Int, Ne*pree)
        sqrtK = sqrt(K)
        tau_meme = 10   # (ms)
        g = 1.0
        je = 2.0 / sqrtK * tau_meme * g
        ji = 2.0 / sqrtK * tau_meme * g 
        jee = 0.15je 
        jei = je 
        jie = -0.75ji 
        jii = -ji
        (edge_dict,w0Weights,Ne,Ni,Lexc,Linh) = potjans_weights(Ncells, jee, jie, jei, jii, ccu, scale)
        nc0,w0Index = build_w0Index(edge_dict,Ncells)
        @save "potjans_matrix.jld2" edge_dict w0Weights Ne Ni Lexc Linh Ncells jee jie jei jii nc0 w0Index
        UnicodePlots.spy(w0Weights) |> display
        if verbose
            UnicodePlots.spy(Lexc) |> display
            UnicodePlots.spy(Linh) |> display    
            println(maximum(Lexc))
            println(minimum(Lexc))StaticWeights-erdos-renyi
        end
        UnicodePlots.spy(Lexc) |> display
        UnicodePlots.spy(Linh) |> display   
        if verbose
    
            println(maximum(Lexc))
            println(minimum(Lexc))Ncells, jee, jie, jei, jii, ccu, scale jee jie jei jii nc0 w0Index

    end
    (edge_dict, w0Weights, Ne, Ni, Lexc, Linh, Ncells, jee, jie, jei, jii, nc0, w0Index) 
end
=#
function genStaticWeights(args)
    # unpack arguments
    #, this is just going through the motions, mostly not used.

    Ncells, jee, jie, jei, jii, ccu, scale = args#map(x->args[x],[:Ncells, :jee, :jie, :jei, :jii, :ccu, :scale])
    if true #!isfile("potjans_matrix.jld2")
        #ccu, scale=1.0::Float64
        #(Ncells::Int64, jee::Float64, jie::Float64, jei::Float64, jii::Float64, ccu, scale=1.0::Float64)
        (edge_dict,w0Weights,Ne,Ni,Lexc,Linh) = potjans_weights(Ncells, jee, jie, jei, jii, ccu, scale)
        nc0,w0Index = build_w0Index(edge_dict,Ncells)

        @save "potjans_matrix.jld2" edge_dict w0Weights Ne Ni Lexc Linh Ncells jee jie jei jii nc0 w0Index

    else
        @load "potjans_matrix.jld2" edge_dict w0Weights Ne Ni Lexc Linh Ncells jee jie jei jii nc0 w0Index

    end
    UnicodePlots.spy(w0Weights) |> display
    if verbose
        UnicodePlots.spy(Lexc) |> display
        UnicodePlots.spy(Linh) |> display    
        println(maximum(Lexc))
        println(minimum(Lexc))
    end
        #StaticWeights-erdos-renyi
    UnicodePlots.spy(Lexc) |> display
    UnicodePlots.spy(Linh) |> display   
    if verbose

        println(maximum(Lexc))
        println(minimum(Lexc))
        println(maximum(Linh))
        println(minimum(Linh))
    end
    return w0Index, w0Weights, nc0
end

function genPlasticWeights(args::Dict{Symbol, Real}, w0Index, nc0, ns0)

    wpWeightFfwd = randn(rng, p.Ncells, p.Lffwd) * wpffwd
    
    return wpWeightFfwd, wpWeightIn, wpIndexIn, ncpIn
end


function as_yet_unused_but_should_be_used_param()
    """
    Again... :This file consists of a function stack that seemed necessary to achieve a network with Potjans like wiring in Julia using TrainSpikeNet.jl to simulate.
    This code draws heavily on the PyNN OSB Potjans implementation code found here:
    https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
    
    Much busy work involved in translating this code Python -> Julia.



    neuron_params = Dict(
        'cm'        : 0.25,  # nF
        'i_offset'  : 0.0,   # nA
        'tau_m'     : 10.0,  # ms
        'tau_refrac': 2.0,   # ms
        'tau_syn_E' : 0.5,   # ms
        'tau_syn_I' : 0.5,   # ms
        'v_reset'   : -65.0,  # mV
        'v_rest'    : -65.0,  # mV
        'v_thresh'  : -50.0  # mVccu, scale=1.0::Float64 2, 'L6': 3)
    n_layers = len(layers)
    pops = Dict('E': 0, 'I': 1)
    n_pops_per_layer = len(pops)
    structure = Dict('L23': Dict('E': 0, 'I': 1),
                'L4' : Dict('E': 2, 'I': 3),
                'L5' : Dict('E': 4, 'I': 5),
                'L6' : Dict('E': 6, 'I': 7))

    # Numbers of neurons in full-scale model
    N_full = Dict(
    'L23': {'E': 20683, 'I': 5834},
    'L4' : {'E': 21915, 'I': 5479},
    'L5' : {'E': 4850, 'I': 1065},
    'L6' : {'E': 14395, 'I': 2948}
    )

    N_E_total = N_full['L23']['E']+N_full['L4']['E']+N_full['L5']['E']+N_full['L6']['E']

    x_dimension = 1000
    z_dimension = 1000
    thalamus_offset = -300

    total_cortical_thickness = 1500.0

    # Have the thicknesses proportional to the numbers of E cells in each layer
    layer_thicknesses = {
    'L23': total_cortical_thickness*N_full['L23']['E']/N_E_total,
    'L4' : total_cortical_thickness*N_full['L4']['E']/N_E_total,
    'L5' : total_cortical_thickness*N_full['L5']['E']/N_E_total,
    'L6' : total_cortical_thickness*N_full['L6']['E']/N_E_total,
    'thalamus' : 100
    }

    establish_connections = True

    # Probabilities for >=1 connection between neurons in the given populations. 
    # The first index is for the target population; the second for the source population
    #             2/3e      2/3i    4e      4i      5e      5i      6e      6i

    # In-degrees for external inputs
    K_ext = {
    'L23': {'E': 1600, 'I': 1500},
    'L4' : {'E': 2100, 'I': 1900},
    'L5' : {'E': 2000, 'I': 1900},
    'L6' : {'E': 2900, 'I': 2100}
    }

    # Mean rates in the full-scale model, necessary for scaling
    # Precise values differ somewhat between network realizations
    full_mean_rates = {
    'L23': {'E': 0.971, 'I': 2.868},
    'L4' : {'E': 4.746, 'I': 5.396},
    'L5' : {'E': 8.142, 'I': 9.078},
    'L6' : {'E': 0.991, 'I': 7.523}
    }

    # Mean and standard deviation of initial membrane potential distribution
    V0_mean = -58.  # mV
    V0_sd = 5.     # mV

    # Background rate per synapse
    bg_rate = 8.  # spikes/s

    # Mean synaptic weight for all excitatory projections except L4e->L2/3e
    w_mean = 87.8e-3  # nA
    w_ext = 87.8e-3  # nA
    # Mean synaptic weight for L4e->L2/3e connections 
    # See p. 801 of the paper, second paragraph under 'Model Parameterization', 
    # and the caption to Supplementary Fig. 7
    w_234 = 2 * w_mean  # nA

    # Standard deviation of weight distribution relative to mean for 
    # all projections except L4e->L2/3e
    w_rel = 0.1
    # Standard deviation of weight distribution relative to mean for L4e->L2/3e
    # This value is not mentioned in the paper, but is chosen to match the 
    # original code by Tobias Potjans
    w_rel_234 = 0.05

    # Means and standard deviations of delays from given source populations (ms)
    d_mean = {'E': 1.5, 'I': 0.75}
    d_sd = {'E': 0.75, 'I': 0.375}

    # Parameters for transient thalamic input
    thalamic_input = False
    thal_params = {
    # Number of neurons in thalamic population
    'n_thal'      : 902,
    # Connection probabilities
    'C'           : {'L23': {'E': 0, 'I': 0},
                    'L4' : {'E': 0.0983, 'I': 0.0619},
                    'L5' : {'E': 0, 'I': 0},
                    'L6' : {'E': 0.0512, 'I': 0.0196}},
    'rate'        : 120.,  # spikes/s;
    'start'       : 700.,  # ms
    'duration'    : 10.   # ms;
    }

    # Plotting parameters
    create_raster_plot = True
    raster_t_min = 0  # ms
    raster_t_max = sim_params.simulator_params[simulator]['sim_duration']  # ms
    # Fraction of recorded neurons to include in raster plot
    frac_to_plot = 0.01
    """
end

