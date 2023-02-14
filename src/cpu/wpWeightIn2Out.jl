function wpWeightIn2Out!(wpWeightOut, Ncells, ncpIn, wpIndexIn, wpIndexConvert, wpWeightIn)
    for postCell = 1:Ncells
        for i = 1:ncpIn[postCell]
            preCell = wpIndexIn[i, postCell]
            postCellConvert = wpIndexConvert[i, postCell]
            wpWeightOut[postCellConvert, preCell] = wpWeightIn[i, postCell]
        end
    end
end
