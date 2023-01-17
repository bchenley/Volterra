class Polynomial(torch.nn.Module):
  def __init__(self,
               units, degree = 1,
               c_init = None, c_train = True,
               include_bias = False,
               device = device, dtype = torch.float32):
    super(Polynomial, self).__init__()

    if c_init is None:
      c_init = torch.normal(size = (degree+int(include_bias), units), mean = 0., std = 0.001).to(device = device, dtype = dtype)
    else:
      c_init = c_init.to(device = device, dtype = dtype)
    c = torch.nn.Parameter(data = c_init, requires_grad = c_train)

    self.c = c
    self.units, self.degree = units, degree
    self.include_bias = include_bias
    self.device, self.dtype = device, dtype 

  def forward(self, X):

    X = X.to(device = self.device, dtype = self.dtype)
    
    pows = torch.arange(1-int(self.include_bias), (self.degree+1)).to(device = self.device, dtype = self.dtype) 
    
    y = torch.zeros_like(X).to(device = self.device, dtype = self.dtype) 

    for h in range(self.units):
      y[:, h:(h+1)] = X[:, h:(h+1)].pow(pows) @ self.c[:, h:(h+1)]

    return y
