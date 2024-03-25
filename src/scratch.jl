Base.@kwdef struct Scratch{MatrixTimeInt, VectorInt, VectorCharge, VectorFloat, VectorCurrent, MatrixCurrent, VectorInvTime, InvTime, VectorFloat64dt, VectorVoltage, VectorNoise}

    times::MatrixTimeInt = Matrix(undef, p.Ncells, maxTimes+extra)  # times of recurrent spikes throughout trial
    ns::VectorInt = Vector(undef, p.Ncells)         # number of recurrent spikes in trial
    timesX::MatrixTimeInt = Matrix(undef, p.LX, maxTimes+extra)     # times of feed-forward spikes throughout trial
    nsX::VectorInt = Vector(undef, p.LX)            # number of feed-forward spikes in trial

    inputsE::VectorCharge = Vector(undef, p.Ncells+extra)      # excitatory synaptic currents to neurons via balanced connections at one time step
    inputsI::VectorCharge = Vector(undef, p.Ncells+extra)      # inhibitory synaptic currents to neurons via balanced connections at one time step
    inputsP::VectorCharge = Vector(undef, p.Ncells+extra)      # synaptic currents to neurons via plastic connections at one time step
    inputsEPrev::VectorCharge = Vector(undef, p.Ncells+extra)  # copy of inputsE from previous time step
    inputsIPrev::VectorCharge = Vector(undef, p.Ncells+extra)  # copy of inputsI from previous time step
    inputsPPrev::VectorCharge = Vector(undef, p.Ncells+extra)  # copy of inputsP from previous time step

    spikes::VectorFloat = Vector(undef, p.Ncells+extra)      # spikes emitted by each recurrent neuron at one time step
    spikesPrev::VectorFloat = Vector(undef, p.Ncells+extra)  # copy of spike from previous time step
    spikesX::VectorFloat = Vector(undef, p.LX)         # spikes emitted by each feed-forward neuron at one time step
    spikesXPrev::VectorFloat = Vector(undef, p.LX)     # copy of spikesX from previous time step

    u_bale::VectorCurrent = Vector(undef, p.Ncells)   # synapse-filtered excitatory current (i.e. filtered version of inputsE)
    u_bali::VectorCurrent = Vector(undef, p.Ncells)   # synapse-filtered inhibitory current (i.e. filtered version of inputsI)
    uX_plas::VectorCurrent = Vector(undef, p.Ncells)  # synapse-filtered plastic current (i.e. filtered version of inputsP)
    u_bal::VectorCurrent = Vector(undef, p.Ncells)    # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)
    u::VectorCurrent = Vector(undef, p.Ncells)        # sum of u_bale and u_bali (i.e. synaptic current from the balanced connections)

    r::VectorInvTime = Vector(undef, p.Ncells+extra)  # synapse-filtered recurrent spikes (i.e. filtered version of spike)
    rX::VectorInvTime = Vector(undef, p.LX)     # synapse-filtered feed-forward spikes (i.e. filtered version of spikesX)

    rrXhistory::CircularArrayBuffer = CircularArrayBuffer{InvTime}(p.LX+p.Ncells+extra, round(Int, p.PHistory / p.learn_every))

    X::VectorCurrent = Vector(undef, p.Ncells)  # total external input to neurons

    lastSpike::VectorFloat64dt = Vector(undef, p.Ncells)  # last time a neuron spiked

    v::VectorVoltage = Vector(undef, p.Ncells)     # membrane voltage
    noise::VectorNoise = Vector(undef, p.Ncells)  # actual noise added at each time step

    rndX::VectorFloat = Vector(undef, p.LX)  # uniform noise to generate Poisson feed-forward spikes

    u_exccell::MatrixCurrent = Matrix(undef, p.Nsteps, p.example_neurons)
    u_inhcell::MatrixCurrent = Matrix(undef, p.Nsteps, p.example_neurons)
    u_bale_exccell::MatrixCurrent = Matrix(undef, p.Nsteps, p.example_neurons)
    u_bali_exccell::MatrixCurrent = Matrix(undef, p.Nsteps, p.example_neurons)
    u_bale_inhcell::MatrixCurrent = Matrix(undef, p.Nsteps, p.example_neurons)
    u_bali_inhcell::MatrixCurrent = Matrix(undef, p.Nsteps, p.example_neurons)
    u_plas_exccell::MatrixCurrent = Matrix(undef, p.Nsteps, p.example_neurons)
    u_plas_inhcell::MatrixCurrent = Matrix(undef, p.Nsteps, p.example_neurons)

    u_rollave::MatrixCurrent = Matrix(undef, learn_nsteps, p.Ncells)
    u_bale_rollave::MatrixCurrent = Matrix(undef, learn_nsteps, p.Ncells)
    u_bali_rollave::MatrixCurrent = Matrix(undef, learn_nsteps, p.Ncells)
    u_plas_rollave::MatrixCurrent = Matrix(undef, learn_nsteps, p.Ncells)
    u_rollave_cnt::VectorInt = Vector(undef, learn_nsteps)

end
