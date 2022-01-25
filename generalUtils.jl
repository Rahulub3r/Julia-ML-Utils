
function kappa(yhat, y)
    # Get confusion matrix
    try
        confmat = MLJ.confusion_matrix(mode.(yhat), y) #probabilistic
    catch
        confmat = MLJ.confusion_matrix(yhat, y) #deteministic
    end
    confmat = confmat.mat

    # sizes
    c = size(confmat)[1] # number of classes
    m = sum(confmat) # number of instances

    # relative observed agreement
    diags = [confmat[i, i] for i in 1:c]
    p_0 = sum(diags)/m

    # probability of agreement due to chance
    # for each class, this would be: (# positive predictions)/(# instances) * (# positive observed)/(# instances)
    p_e = 0
    for i in 1:c
        p_e_i = sum(confmat[i, j] for j in 1:c) * sum(confmat[j, i] for j in 1:c)/m^2
        p_e += p_e_i
    end

    # Kappa calculation
    κ = (p_0 - p_e)/(1 - p_e)

    return κ
end