class HermiteModulator(torch.nn.Module):
  def __init__(self,
               t_domain,
               scale = True,
               kind = 1,
               degree = 1,
               include_bias = True,
               device = device, dtype = torch.float32):
    super(HermiteModulator, self).__init__()

    self.degree = degree
    self.include_bias = include_bias
    self.num_modulators = degree + int(include_bias)

    self.device, self.dtype = device, dtype 

    self.t_domain = t_domain.to(device = device, dtype = dtype)
    self.scale = scale
    self.kind = kind
    self.functions = self.generate_basis_functions()

  def generate_basis_functions(self):
    
    N = len(self.t_domain)
    
    y = torch.zeros((N, (self.degree+1))).to(device = self.device, dtype = self.dtype)
    
    t = self.t_domain 

    t = t / (t.max() - t.min()) if self.scale else t

    for q in range(0, (self.degree+1)):
      if q == 0: 
        y[:, 0] = torch.ones((N,)).to(device = self.device, dtype = self.dtype)
      elif q == 1:              
        y[:, 1:2] = self.kind*t*y[:, 0:1]
      else:
        y[:, q:(q+1)] = self.kind*(t*y[:, (q-1):q] - (q-1)*y[:, (q-2):(q-1)])
        
    if not self.include_bias:
      y = y[:,1:]

    return y

  def forward(self, X, t):
    
    X = X.to(device = self.device, dtype = self.dtype)
    t = t.to(device = self.device, dtype = self.dtype)

    N = X.shape[0]
  
    _, idx, _ = np.intersect1d(self.t_domain.cpu().numpy(), t.cpu().numpy(), return_indices = True)

    y = X*self.functions[idx]

    return y
