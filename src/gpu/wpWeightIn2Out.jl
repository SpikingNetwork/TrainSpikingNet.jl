function wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    function kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        @inbounds for i=i0:istride:size(wpWeightIn,1)
            for j=j0:jstride:size(wpWeightIn,2)
                wpWeightOut[wpIndexConvert[i,j],wpIndexIn[i,j]] = wpWeightIn[i,j]
            end
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)
    config = launch_configuration(kernel.fun)
    dims = size(wpWeightIn)
    xthreads = min(32, dims[1])
    ythreads = min(fld(config.threads, xthreads), cld(prod(dims), xthreads))
    xblocks = min(config.blocks, cld(dims[1], xthreads))
    yblocks = min(cld(config.blocks, xblocks), cld(dims[2], ythreads))
    kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn;
           threads=(xthreads,ythreads), blocks=(xblocks<<2,yblocks<<2))
end
