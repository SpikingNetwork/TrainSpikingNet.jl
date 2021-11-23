function convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    @static if p.PType == Array
        wpWeightOut[CartesianIndex.(wpIndexConvert,wpIndexIn)] = permutedims(wpWeightIn, (3,1,2))
    else
        wpWeightOut[CartesianIndex.(wpIndexConvert,wpIndexIn)] = permutedims(wpWeightIn, (2,1))
    end
    return wpWeightOut
end
