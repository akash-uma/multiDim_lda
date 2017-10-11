function [ lowd_x, projMat ] = multiDim_lda( X, y, nDims )

    [nSamp,xDim] = size(X);
    if nDims > xDim-1
        error('nDims must be less than data dimensionality');
    end

    lowd_x = zeros(nSamp,nDims);
    projMat = zeros(xDim,nDims);
    
    newX = X;
    for iDim = 1:nDims
        params(iDim) = train_lda(newX,y);
        newX = newX*params(iDim).projMat(:,2:end);
        lowd_x(:,iDim) = params(iDim).projData;
        if iDim==1
            currProjVec = params(iDim).projVec;
        else
            currProjVec = params(iDim).projVec;
            for ii = (iDim-1):-1:1
                currProjVec = params(ii).projMat(:,2:end)*currProjVec;
            end
        end
        projMat(:,iDim) = currProjVec;
    end
    
end

