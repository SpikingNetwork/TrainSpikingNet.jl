function convertWgtIn2Out(p,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    wpWeightOut[CartesianIndex.(wpIndexConvert,wpIndexIn)] = permutedims(wpWeightIn, (3,1,2))
    return wpWeightOut
end
