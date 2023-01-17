class FourierModulator(torch.nn.Module):
  def __init__(self,
               num_freqs,
               sampling_freq = 1,
               f_init = None, f_train = True,
               p_init = None, p_train = True,               
               device = device, dtype = torch.float32):
    super(FourierModulator, self).__init__()

    if f_init is None:
      f_init = (sampling_freq/4)*torch.ones(size = (1, num_freqs)).to(device = device, dtype = dtype)
    else:
      f_init = f_init.to(device = device, dtype = dtype)

    if p_init is None:
      p_init = torch.zeros(size = (1, num_freqs)).to(device = device, dtype = dtype)
    else:
      p_init = p_init.to(device = device, dtype = dtype)      

    f = torch.nn.Parameter(data = f_init, requires_grad = f_train)
    p = torch.nn.Parameter(data = p_init, requires_grad = p_train)
    
    self.f, self.p = f, p
    self.num_modulators = num_freqs
    self.sampling_freq = sampling_freq
    self.device, self.dtype = device, dtype 
    
  def forward(self, t):
    
    t = t.to(device = self.device, dtype = self.dtype)
    
    y = torch.cos(2*torch.pi*self.f*t + self.p)
    
    return y
