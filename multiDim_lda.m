function [ lowd_x, projMat, dPrimes ] = multiDim_lda( X, y, nDims )

    [nSamp,xDim] = size(X);
    if nDims > xDim-1
        error('nDims must be less than data dimensionality');
    end

    lowd_x = zeros(nSamp,nDims);
    projMat = zeros(xDim,nDims);
    dPrimes = zeros(1,nDims);
    
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
        proj1 = lowd_x(y==params.classLabels(1));
        proj2 = lowd_x(y==params.classLabels(2));
        dPrimes(iDim) = abs(mean(proj1)-mean(proj2))/sqrt((var(proj1,1)+var(proj2,1))/2);
    end
    
end

