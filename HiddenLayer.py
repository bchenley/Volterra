class HiddenLayer(torch.nn.Module):
  def __init__(self,
               input_dim, output_dim = 1, 
               # inbound weights
               w_init = None, w_train = True,
               # activation
               activation = 'polynomial',
               # polynomial activaation
               degree = 1,
               c_init = None, c_train = True,
               # sigmoidal activation
               s_init = None, s_train = True,
               bs_init = None, bs_train = True,
               # bias
               b_init = None, b_train = False,
               device = device, dtype = torch.float32):
    
    super(HiddenLayer, self).__init__()
    
    if w_init is None:
      w_init = torch.normal(size = (input_dim, output_dim), mean = 0., std = 0.001)
    w = torch.nn.Parameter(data = w_init.to(device = device, dtype = dtype), requires_grad = w_train)

    F = None
    if activation == 'polynomial':
      if c_init is None:
        c_init = torch.normal(size = (degree, output_dim), mean = 0., std = 0.001)

        if degree == 1:
          c_init = torch.ones(size = (1, output_dim))
          c_train = False
      
      F = Polynomial(units = output_dim, degree = degree,
                     c_init = c_init, c_train = c_train, 
                     include_bias = False, 
                     device = device, dtype = dtype)

    elif activation == 'sigmoid':

      F = Sigmoidal(units = output_dim, degree = degree,
                    s_init = s_init, s_train = s_train, 
                    b_init = bs_init, b_train = bs_train, 
                    device = device, dtype = dtype)
    
    elif activation == 'softmax':

      F = Softmax(units = output_dim, 
                  device = device, dtype = dtype)
    
    b = None    
    if b_train:
      if b_init is None:
        b_init = torch.zeros(size = (1, output_dim)).to(device = device, dtype = dtype)
      b = torch.nn.Parameter(data = b_init, requires_grad = b_train)
    
    self.input_dim, self.output_dim, self.degree = input_dim, output_dim, degree
    self.w_init, self.w_train, self.w = w_init, w_train, w
    self.b_init, self.b_train, self.b = b_init, b_train, b
    
    self.activation, self.F = activation, F
    
    self.device, self.dtype = device, dtype

  def forward(self, X):

    X = X.to(device = self.device, dtype = self.dtype) 

    y = X @ self.w

    if self.F is not None: y = self.F(y) 

    if self.b is not None: y += self.b
    
    return y
