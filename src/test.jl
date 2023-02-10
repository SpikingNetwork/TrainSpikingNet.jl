using Pkg;  Pkg.activate(dirname(@__DIR__), io=devnull)

using TrainSpikingNet, ArgParse

function ArgParse.parse_item(::Type{Vector{Int}}, x::AbstractString)
    return eval(Meta.parse(x))
end

aps = ArgParseSettings()

@add_arg_table! aps begin
    "--ntrials", "-n"
        help = "number of repeated trials to average over"
        arg_type = Int
        default = 1
        range_tester = x->x>0
    "--ineurons_to_test", "-i"
        help = "which neurons to test"
        arg_type = Vector{Int}
        default = collect(1:16)
        range_tester = x->all(x.>0)
    "--restore_from_checkpoint", "-r"
        help = "use checkpoint R.  default is to use the last one"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "--no-plot"
        help = "just save to JLD2 file"
        action = :store_true
    "--gpu", "-g"
        help = "use the GPU"
        action = :store_true
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
end

parsed_args = parse_args(aps)

config(parsed_args["data_dir"], parsed_args["gpu"] ? :gpu : :cpu)

test(; ntrials = parsed_args["ntrials"],
       ineurons_to_test = parsed_args["ineurons_to_test"],
       restore_from_checkpoint = parsed_args["restore_from_checkpoint"],
       no_plot = parsed_args["no-plot"])
