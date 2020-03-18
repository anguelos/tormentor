import torch


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
