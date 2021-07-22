function funRollingAvg(startInd,endInd,movavg,cnt,x)
    movavg[startInd:endInd,:] .+= transpose(x)
    cnt[startInd:endInd,:] .+= 1
end
