import torch


def fast_gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, epsilon: float, norm: str = "2",
                         loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack on the input x.
    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is the number of classes.
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the number of channels, and N is the image
       dimension.
       The input batch of images. Note that x.requires_grad must have been active before computing the logits
       (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation (before clipping) will have a norm of
        exactly epsilon as measured by the desired norm (see argument: norm).
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", the perturbation (before clipping)
         will have a L_1 norm of exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.

    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    

    loss = loss_fn(logits, y)

    gradient = torch.autograd.grad(loss, x)[0]

    avoid_zero_div = torch.tensor(1e-12, requires_grad=True).to(y.device)
    
    dims = tuple(range(1, len(gradient.shape)))
    
    l1_norm_manual = torch.max(avoid_zero_div, torch.sum(torch.abs(gradient), dims, keepdim=True))
    
    l2_norm_manual = torch.sqrt(torch.max(avoid_zero_div, torch.sum(gradient ** 2, dims, keepdim=True)))


    delta = {
        "1": gradient / l1_norm_manual,
        "2": gradient / l2_norm_manual,
        "inf": torch.sign(gradient)
    }

    x_pert = torch.clamp(x + epsilon * delta[norm], 0, 1)

   
    return x_pert.detach()
