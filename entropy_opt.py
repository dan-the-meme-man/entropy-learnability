import torch
from torch import nn
from torch import Tensor

class MSEAgainstEntropyLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, dist: Tensor, true_entropy: Tensor) -> Tensor:
        
        dist = torch.softmax(dist, dim=0)
        
        approx_entropy = -(dist * dist.log()).sum()
        
        mse = (approx_entropy - true_entropy) ** 2
        
        return mse
    
class MSEAgainstEntropyAndVarEntropyLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, dist: Tensor, true_entropy: Tensor, true_varent: Tensor) -> Tensor:
        
        dist = torch.softmax(dist, dim=0)
        
        X = -dist.log()
        approx_entropy = (dist * X).sum()
        E_X_sq = (dist * X * X).sum()
        approx_varent = E_X_sq - (approx_entropy ** 2)
        
        mse = 0.5 * (approx_entropy - true_entropy) ** 2 + 0.5 * (approx_varent - true_varent) ** 2
        
        return mse
    
def get_dist(
    criterion: nn.Module,
    vocab_size: int,
    desired_entropy: float,
    desired_varent: float = None,
    do_logging: bool = True,
    tol: float = 1e-6
) -> tuple[Tensor, float, float]:
    
    losses = []
    
    if desired_varent is not None:
        msg = 'Cannot specify a varentropy and use a criterion that ignores it!'
        assert isinstance(criterion, MSEAgainstEntropyAndVarEntropyLoss), msg
    if isinstance(criterion, MSEAgainstEntropyLoss):
        msg = 'Cannot specify a varentropy and use a criterion that ignores it!'
        assert desired_varent is None
    if isinstance(criterion, MSEAgainstEntropyAndVarEntropyLoss):
        msg = 'Must specify a varentropy and use a criterion that includes it!'
        assert desired_varent is not None, msg
    
    dist = nn.Parameter(torch.randn((vocab_size,), dtype=torch.float64))
    dist.requires_grad = True
    DE = torch.tensor(desired_entropy, dtype=torch.float64)
    DE.requires_grad = False
    if desired_varent is not None:
        DV = torch.tensor(desired_varent, dtype=torch.float64)
        DV.requires_grad = False
    
    optimizer = torch.optim.AdamW([dist], lr=0.001)
    
    i = 0
    if do_logging:
        print('-----------------------------------------------------')
    while True:
        
        optimizer.zero_grad()
        
        if desired_varent is None:
            loss = criterion(dist, DE)
        else:
            loss = criterion(dist, DE, DV)
        loss.backward()
        optimizer.step()
        
        if (i % 1000 == 0):
            with torch.no_grad():
                loss_val = loss.item()
                if do_logging:
                    print(f'loss: {loss_val:.4}')
                if loss_val < tol:
                    break
                losses.append(loss_val)
                if len(losses) > 1:
                    if abs(losses[-1] - losses[-2]) < tol:
                        if losses[-1] >= tol:
                            msg = 'Optimization did not converge!'
                            Warning(msg)
                        break

        i += 1
        
        if i > 100_000:
            msg = 'Optimization did not converge!'
            Warning(msg)
            break
        
        
    
    final_dist = torch.softmax(dist, dim=0)
    
    X = -final_dist.log()
    E_X = (final_dist * X).sum()
    E_X_sq = (final_dist * X * X).sum()
    mean = E_X.item()
    var = E_X_sq.item() - (mean ** 2)
    
    if do_logging:
        print('-----------------------------------------------------')
        print(f'sum of probabilities (should be 1): {final_dist.sum()}')
        print('-----------------------------------------------------')
        print(f'desired entropy:    {desired_entropy}')
        print(f' actual entropy:    {mean}')
        print('-----------------------------------------------------')
        print(f'desired varentropy: {desired_varent}')
        print(f' actual varentropy: {var}')
        print('-----------------------------------------------------')
    return final_dist, mean, var

"""
    Example calls:
    
    ent_crit = MSEAgainstEntropyLoss()
    
    returned_dist = get_dist(
        criterion       = ent_crit, # must match
        vocab_size      = 20,
        desired_entropy = 1.6,
        desired_varent  = None, # must match
        do_logging      = True,
        tol             = 1e-6
    )
    
    ent_var_crit = MSEAgainstEntropyAndVarEntropy()
    
    returned_dist = get_dist(
        criterion       = ent_var_crit, # must match
        vocab_size      = 20,
        desired_entropy = 1.6,
        desired_varent  = 0.3, # must match
        do_logging      = True,
        tol             = 1e-6
    )

"""