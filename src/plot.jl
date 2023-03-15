using Pkg;  Pkg.activate(dirname(@__DIR__), io=devnull)

using TrainSpikingNet, ArgParse

# --- define command line arguments --- #
function ArgParse.parse_item(::Type{Vector{Int}}, x::AbstractString)
    return eval(Meta.parse(x))
end

aps = ArgParseSettings()

@add_arg_table! aps begin
    "--ineurons_to_plot", "-i"
        help = "which neurons to plot.  must be the same or a subset of ineurons_to_test used in test.jl"
        arg_type = Vector{Int}
        default = collect(1:16)
        range_tester = x->all(x.>0)
    "test_file"
        help = "full path to the JLD file output by test.jl.  this same directory needs to contain the parameters in param.jld2, the synaptic targets in utarg.jld2, and (optionally) the spike rate in rate.jld2"
        required = true
end

parsed_args = parse_args(aps)

plot(parsed_args["test_file"],
     ineurons_to_plot = parsed_args["ineurons_to_plot"],
     load_init_code=true)
