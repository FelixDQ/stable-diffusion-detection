import torch 

def tfft(x):
    return torch.stack(
                [
                    torch.fft.fft2(x[0]),
                    torch.fft.fft2(x[1]),
                    torch.fft.fft2(x[2]),
                ]
            )

def tabs(x):
    return torch.abs(x)

def tlog(x):
    return torch.log10(x)

def tnorm(x):
    return ((x - x.min()) / (x.max() - x.min()) * 2) - 1

def spec(x):
    return tnorm(tfft(x).abs().log())
    