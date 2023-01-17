class LaguerreFilterbank(torch.nn.Module):
  def __init__(self, 
               num_filters = 3, 
               a_init = 0.5, a_train = True,
               dt = 1,
               dtype = torch.float32, device = device):
    super(LaguerreFilterbank, self).__init__()

    a_init = torch.tensor(a_init).to(device = device, dtype = dtype)

    a = torch.nn.Parameter(a_init, requires_grad = a_train)
    
    self.num_filters = num_filters
    self.a_train = a_train
    self.a = a
    self.dt = dt

    self.dtype, self.device = dtype, device

  def forward(self, X, v = None):
    
    N = X.shape[0]
    V = torch.zeros((N, self.num_filters)).to(device = self.device, dtype = self.dtype)

    if v is None:
      v = torch.zeros((1, self.num_filters)).to(device = self.device, dtype = self.dtype)

    for n in range(N):

      # 0th order DLF
      V[n:(n+1), 0:1] = torch.sqrt(self.a)*v[:, 0:1] + self.dt*torch.sqrt(1-self.a)*X[n:(n+1)]
      #
      
      # ith order DLFs
      for j in range(1, self.num_filters):    
        v_i_j = torch.sqrt(self.a)*(v[:,j:(j+1)] + V[n:(n+1),(j-1):j]) - v[:,(j-1):j]
        V[n:(n+1), j:(j+1)] = v_i_j
      #
    
      v = V[n:(n+1)]

    return V, v

  def basis(self):

    with torch.inference_mode(): 

      b = torch.empty((0, self.num_filters)).to(device = self.device, dtype = self.dtype)
      v = torch.zeros((1, self.num_filters)).to(device = self.device, dtype = self.dtype)
       
      v_n, _ = self(X = torch.ones((1, 1)).to(device = self.device, dtype = self.dtype), 
                    v = v)
      
      while (v_n[-1, :].abs() > 1e-4).any():      
        b = torch.cat((b, v_n), axis = 0)
        
        v_n, _ = self(X = torch.zeros((1, 1)).to(device = self.device, dtype = self.dtype), 
                      v = v_n) 
        
    return b

  def conv(self, X):

    if type(X) is not torch.Tensor:
      X = torch.tensor(X).to(device = self.device, dtype = self.dtype)

    b = self.basis()
    
    M = b.shape[0]
    X_ = X.T.unsqueeze(0)
    b_ = b.flip(dims = [0]).T.unsqueeze(dim = 1)         
    V =  torch.nn.functional.conv1d(X_, b_,
                                    bias=None, 
                                    stride=1, padding = M-1, 
                                    dilation=1, groups=1).mT.reshape(-1, self.num_filters)[:X.shape[0], :]

    return V
    
