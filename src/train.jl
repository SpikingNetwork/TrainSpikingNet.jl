using Pkg;  Pkg.activate(dirname(@__DIR__), io=devnull)

using TrainSpikingNet, ArgParse

aps = ArgParseSettings()

@add_arg_table! aps begin
    "--nloops", "-n"
        help = "number of iterations to train"
        arg_type = Int
        default = 1
    "--correlation_interval", "-c"
        help = "measure correlation every C training loops.  default is every loop"
        arg_type = Int
        default = 1
        range_tester = x->x>0
    "--save_best_checkpoint", "-s"
        help = "save the learned weights and covariance matrices with the highest measured correlation too.  default is to only save the last one"
        action = :store_true
    "--restore_from_checkpoint", "-r"
        help = "continue training from checkpoint R.  default is to start from the beginning"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "--monitor_resources_used", "-m"
        help = "measure power, cores, and memory usage every R seconds.  default is never"
        arg_type = Int
        default = nothing
        range_tester = x->x>0
    "--gpu", "-g"
        help = "use the GPU"
        action = :store_true
    "data_dir"
        help = "full path to the directory containing the parameters file"
        required = true
end

parsed_args = parse_args(aps)

config(parsed_args["data_dir"], parsed_args["gpu"] ? :gpu : :cpu)

train(; nloops = parsed_args["nloops"],
        correlation_interval = parsed_args["correlation_interval"],
        save_best_checkpoint = parsed_args["save_best_checkpoint"],
        restore_from_checkpoint = parsed_args["restore_from_checkpoint"],
        monitor_resources_used = parsed_args["monitor_resources_used"])
