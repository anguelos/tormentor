import torch


def debug_pattern(num_rows:int, num_cols:int,square_size:int)->torch.FloatTensor:
    res_img = torch.zeros([num_cols*square_size,num_rows*square_size])
    square = (torch.arange(square_size).view(1,square_size) * torch.arange(square_size).view(square_size,1))
    square = square / float(square_size * square_size)
    for y in range(0,num_rows):
        for x in range(y%2, num_cols,2):
            res_img[x*square_size:(x+1)*square_size, y*square_size:(y+1)*square_size] = square
    return res_img.unsqueeze(dim=0).unsqueeze(dim=0)


def PCA(X, k, center=True, scale=False):
    n, p = X.size()
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    X_center = torch.mm(H.double(), X.double())
    covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
    scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
    eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
    components = (eigenvectors[:, :k]).t()
    explained_variance = eigenvalues[:k, 0]
    return {'X': X, 'k': k, 'components': components, 'explained_variance': explained_variance}
