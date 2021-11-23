function convertWgtIn2Out(Ncells,ncpIn,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    for postCell = 1:Ncells
        for i = 1:ncpIn[postCell]
            preCell = wpIndexIn[postCell,i]
            postCellConvert = wpIndexConvert[postCell,i]
            wpWeightOut[postCellConvert,preCell] = wpWeightIn[i,postCell]
        end
    end

    return wpWeightOut
    
end
