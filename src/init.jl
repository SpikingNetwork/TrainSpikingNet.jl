using Pkg;  Pkg.activate(dirname(@__DIR__), io=devnull)

using TrainSpikingNet, ArgParse

function ArgParse.parse_item(::Type{Vector{Int}}, x::AbstractString)
    return eval(Meta.parse(x))
end

aps = ArgParseSettings()

@add_arg_table! aps begin
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
    "--itasks", "-t"
        help = "a vector specifying which tasks to learn"
        arg_type = Vector{Int}
        default = [1]
        range_tester = x->all(x.>0)
end

add_arg_group!(aps, "mutually exclusive arguments.  if neither is specified, sinusoids\nwill be generated for synpatic inputs", exclusive = true);

@add_arg_table! aps begin
    "--utarg_file", "-u"
        help = "full path to the JLD file containing the synaptic current targets"
    "--spikerate_file", "-s"
        help = "full path to the JLD file containing the spike rates"
end

parsed_args = parse_args(aps)

param(parsed_args["data_dir"])

config(parsed_args["data_dir"], :cpu)

init(itasks = parsed_args["itasks"],
     utarg_file = parsed_args["utarg_file"],
     spikerate_file = parsed_args["spikerate_file"])
