using SparseArrays
#using ProgressMeter
using UnicodePlots
using JLD2


"""
This file consists of a function stack that seemed necessary to achieve a network with Potjans like wiring in Julia using TrainSpikeNet.jl to simulate.
This code draws heavily on the PyNN OSB Potjans implementation code found here:
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
"""

function potjans_params(scale_N_cells=1.0)
    """
    Hard coded stuff.
    Outputs adapted Potjans parameters.
    """
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
    ccu = Dict("23E"=>20683, 
                "4E"=>21915, 
                "5E"=>4850, 
                "6E"=>14395, 
                "23I"=>5834,
                "4I"=>5479,
                "5I"=>1065,
                "6I"=>2948)

    # hard coded stuff is manipulated below:

    conn_probs_= copy(conn_probs)
    columns_conn_probs = [col for col in eachcol(conn_probs)][1]
    
    ## this list gets reorganised to reflect top excitatory bottom inhibitory.

    ## Rearrange the whole matrix so that excitatory connections form a top partition 
    and inhibitory neurons form a bottom partition.
    transformed_layer_names = []

    
    for (i,j) in transform_matrix_ind
        # todo use the view syntax here.
        conn_probs_[i,:] = views(conn_probs,j,:)
        append!(transformed_layer_names,layer_names[j])

    end
    for (i,j) in transform_matrix_ind
        # todo use the view syntax here.
        conn_probs_[:,i] = views(conn_probs,:,j)


    end
    ccu = Dict((k,ceil(Int64,v/scale_N_cells)) for (k,v) in pairs(ccu))
    cumulative = Dict() 
    v_old=1
    for (k,v) in pairs(ccu)
        ## A cummulative cell count
        cumulative[k]=collect(v_old:v+v_old)
        v_old=v+v_old
    end    
    return (cumulative,ccu,transformed_layer_names,columns_conn_probs,conn_probs_)
end
function build_index(cumulative::Dict{Any, Any}, conn_probs::Vector{Vector{Float64}})
    """
    Build a nested iterator ideally this will flatten the readability of subsequent code.
    """

    tuple_index = []
    Lexc_ind = []
    Linh_ind = []

    @inbounds for (i,(k,v)) in enumerate(pairs(cumulative))
        @inbounds for src in v
            @inbounds for (j,(k1,v1)) in enumerate(pairs(cumulative))
                @inbounds for tgt in v1
                    if src!=tgt
                        prob = conn_probs[i][j]
                        if rand()<prob
                            push!(tuple_index,(Int64(src),Int64(tgt),k,k1))
                        end
                    end
                end
            end
            if i<round(length(cumulative)/2.0)
                append!(Lexc_ind,src)
            else
                append!(Linh_ind,src)
            end

        end

    end
    
    return tuple_index,Lexc_ind,Linh_ind


end



function potjans_weights(Ncells::Int64, jee::Float64, jie::Float64, jei::Float64, jii::Float64)
    (cumulative,ccu,layer_names,_,conn_probs) = potjans_params()    
    w_mean = 87.8e-3  # nA
    ###
    # Lower memory footprint motivations.
    # a sparse matrix can be stored as a smaller dense matrix.
    # A 2D matrix should be stored as 1D matrix of srcs,tgts
    # A 2D weight matrix should be stored as 1 matrix, which is redistributed in loops using 
    # the 1D matrix of srcs,tgts.
    ###
    Ncells = sum([i for i in values(ccu)]) + 1
    

    w0Index = spzeros(Int,Ncells,Ncells)
    w0Weights = spzeros(Float32,Ncells,Ncells)
    edge_dict = Dict() 
    for src in 1:p.Ncells
        edge_dict[src] = Int64[]
       
    end

    Ne = 0 
    Ni = 0
    tuple_index,Lexc,Linh = build_index(cumulative,conn_probs)
    @inbounds for (src,tgt,k,k1) in tuple_index

        if occursin("E",k) 
            if occursin("E",k1)          
                w0Weights[tgt,src] = jee#/20.
            else# meaning if the same as a logic: occursin("I",k1) is true                   
                w0Weights[tgt,src] = jei#*2.0
            end
            Ne+=1	
        else
        # meaning meaning if the same as a logic: elseif occursin("I",k) is true  
            if occursin("E",k1)                    
                w0Weights[tgt,src] = -jie# *10.0 
            else# eaning meaning if the same as a logic: if occursin("I",k1)      is true               
                w0Weights[tgt,src] = -jii#/2.0  
            end
            Ni+=1

        end
        append!(edge_dict[src],tgt)
            # w0Index building moved to another function w0Index[tgt,src] = tgt

    end
    Lexc = w0Weights[Lexc_ind,:]
    Linh = w0Weights[Linh_ind,:]
    #end
    ##
    # TODO make this commented out call work!
    ##
    #(w0Weights,w0Index) = repartition_exc_inh(w0Index,w0Weights,layer_names)
    return (edge_dict,w0Weights,Ne,Ni,Lexc,Linh)
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

function genStaticWeights(args::Dict{Symbol, Real})
    #
    # unpack arguments, this is just going through the motions, mostly not used.
    Ncells, _, _, _, _, _, jee, jie, jei, jii = map(x->args[x],
            [:Ncells, :Ne, :pree, :prie, :prei, :prii, :jee, :jie, :jei, :jii])

    (edge_dict,w0Weights,Ne,Ni,Lexc,Linh) = potjans_weights(Ncells, jee, jie, jei, jii)
    nc0,w0Index = build_w0Index(edge_dict,Ncells)
    return w0Index, w0Weights, nc0
end

function genPlasticWeights(args::Dict{Symbol, Real}, w0Index, nc0, ns0)
    #
    # unpack arguments, this is just going through the motions, mostly not used.
    Ncells, _, _, _, _, _, Lffwd, wpee, wpie, wpei, wpii, wpffwd = map(x->args[x],
    [:Ncells, :frac, :Ne, :L, :Lexc, :Linh, :Lffwd, :wpee, :wpie, :wpei, :wpii, :wpffwd])
    (edge_dict,wpWeightIn,Ne,Ni,Lexc,Linh) =  potjans_weights(Ncells, wpee, wpie, wpei, wpii)
    ##
    # nc0Max is the maximum number of post synaptic targets
    # its a limit on the outdegree.
    # if this is not known upfront it can be calculated on the a pre-exisiting adjacency matrix as I do below.
    ##
    ncpIn,wpIndexIn = build_w0Index(edge_dict,Ncells)
    wpWeightFfwd = randn(rng, p.Ncells, p.Lffwd) * wpffwd
    
    return wpWeightFfwd, wpWeightIn, wpIndexIn, ncpIn
end


#include("src/genWeightsPotjans.jl")

##
# The following code just gives me the cell count (NCell) upfront, which I need to size arrays in the methods.
##
function get_Ncell()
	ccu = Dict("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)
	ccu = Dict((k,ceil(Int64,v)) for (k,v) in pairs(ccu))
	Ncells = sum([i for i in values(ccu)])+1
	Ne = sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]])
    Ncells, Ne 

end




if !isfile("potjans_matrix.jld2")
	Ncells,Ne = get_Ncell()
    pree = 0.1
    K = round(Int, Ne*pree)
    sqrtK = sqrt(K)
    tau_meme = 10   # (ms)
    g = 1.0
    je = 2.0 / sqrtK * tau_meme * g
    ji = 2.0 / sqrtK * tau_meme * g 
    jx = 0.08 * sqrtK * g 
    jee = 0.15je 
    jie = je 
    ei = -0.75ji 
    jii = -ji

    (edge_dict,w0Weights,Ne,Ni,Lexc,Linh) = potjans_weights(Ncells, jee, jie, jei, jii)
    nc0,w0Index = build_w0Index(edge_dict,Ncells)
    @save "potjans_matrix.jld2" edge_dict w0Weights Ne Ni Lexc Linh Ncells jee jie jei jii nc0 w0Index
else
	@load "potjans_matrix.jld2" edge_dict w0Weights Ne Ni Lexc Linh Ncells jee jie jei jii nc0 w0Index
end