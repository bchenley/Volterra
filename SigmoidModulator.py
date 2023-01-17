class SigmoidModulator(torch.nn.Module):
  def __init__(self,
               num_sigmoids,
               s_init = None, s_train = True,               
               b_init = None, b_train = True,               
               device = device, dtype = torch.float32):
    super(SigmoidModulator, self).__init__()

    if s_init is None:
      s_init = torch.normal(size = (1, num_sigmoids), mean = 0., std = 0.001).to(device = device, dtype = dtype)
    else:
      s_init = s_init.to(device = device, dtype = dtype)
    s = torch.nn.Parameter(data = s_init, requires_grad = s_train)

    if b_init is None:
      b_init = torch.normal(size = (1, num_sigmoids), mean = 0., std = 0.001).to(device = device, dtype = dtype)
    else:
      b_init = b_init.to(device = device, dtype = dtype)
    b = torch.nn.Parameter(data = b_init, requires_grad = b_train)

    self.num_modulators = num_sigmoids
    self.s, self.b = s, b
    self.device, self.dtype = device, dtype 
    
  def forward(self, X, t):
    
    t = t.to(device = self.device, dtype = self.dtype)
    
    y = X/(1 + torch.exp(-self.s*(t - self.b)))
    
    return y
