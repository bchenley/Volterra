class ModulationLayer(torch.nn.Module):
  def __init__(self,
               t_domain, input_dim,
               # legendre
               degree_legendre = None,
               # chebychev
               chebychev_legendre = None,               
               # hermite
               hermite_legendre = None,
               # fourier 
               num_freqs = None,
               f_init = None, f_train = True,
               p_init = None, p_train = True,
               # sigmoid 
               num_sigmoids = None,
               s_init = None, s_train = True,
               bs_init = None, bs_train = True,
               #
               w_init = None, w_train = True,
               b_init = None, b_train = True,
               #  c_init = None, c_train = True,
               device = device, dtype = torch.float32):
    super(ModulationLayer, self).__init__()

    num_modulators = 1
    if type(t_domain) is not torch.Tensor:
      t_domain = torch.tensor(t_domain).view(-1,1).to(device = device, dtype = dtype)

    modulators = torch.nn.ModuleList([])
    F_legendre = None
    if (degree_legendre is not None) & (t_domain is not None):      
      F_legendre = LegendreModulator(t_domain = t_domain,   
                                     scale = True,                      
                                     degree = degree_legendre,
                                     include_bias = False,
                                     device = device, dtype = dtype)
      modulators.append(F_legendre)
      num_modulators += F_legendre.num_modulators

    F_chebychev = None
    if (chebychev_legendre is not None) & (t_domain is not None):
      F_chebychev = ChebychevModulator(t_domain = t_domain,
                                       kind = 1,                              
                                       degree = chebychev_legendre,
                                       include_bias = False,
                                       device = device, dtype = dtype)
      modulators.append(F_chebychev)
      num_modulators += F_chebychev.num_modulators

    F_hermite = None
    if (hermite_legendre is not None) & (t_domain is not None):
      F_hermite = HermiteModulator(t_domain = t_domain,
                                   kind = 1,                              
                                   degree = hermite_legendre,
                                   include_bias = False,
                                   device = device, dtype = dtype)
      modulators.append(F_hermite)
      num_modulators += F_hermite.num_modulators

    F_fourier, f, p = None, None, None
    if num_freqs is not None:
      F_fourier = FourierModulator(num_freqs = num_freqs, 
                                   sampling_freq = 1/dt, 
                                   f_init = f_init, f_train = f_train, 
                                   p_init = p_init, p_train = p_train,
                                   device = device, dtype = torch.float32)
      modulators.append(F_fourier)
      num_modulators += F_fourier.num_modulators

    F_sigmoid, s, bs = None, None, None
    if num_sigmoids is not None:
      F_sigmoid = SigmoidModulator(num_sigmoids = num_sigmoids, 
                                   s_init = s_init, s_train = s_train, 
                                   b_init = bs_init, b_train = bs_train, 
                                   device = device, dtype = dtype) 
      modulators.append(F_sigmoid)
      num_modulators += F_sigmoid.num_modulators

    if w_init is None:
      w_init = torch.normal(size = (input_dim, num_modulators), mean = 0., std = 0.001)
    w = torch.nn.Parameter(data = w_init.to(device = device, dtype = dtype), requires_grad = w_train)

    b = None
    if b_init is None:
      b_init = torch.zeros(size = (1, num_modulators))
    b = torch.nn.Parameter(data = b_init.to(device = device, dtype = dtype), requires_grad = b_train)

    self.input_dim = input_dim
    
    self.modulators, self.num_modulators = modulators, num_modulators
    
    self.w_init, self.w_train, self.w = w_init, w_train, w  
    self.b_init, self.b_train, self.b = b_init, b_train, b  
    
    self.device, self.dtype = device, dtype

  def forward(self, X, t):
    
    X = X.to(device = self.device, dtype = self.dtype)
    t = t.to(device = self.device, dtype = self.dtype)

    N, _ = X.shape

    Z = X @ self.w 
    Z += self.b if self.b is not None else Z

    y = torch.ones((N, 1)).to(device = self.device, dtype = self.dtype)*Z[:, 0:1]

    i = 0
    for m in self.modulators:        
      y = torch.cat((y, m(Z[:, i:(i+m.num_modulators)], t)), 1)
      i += m.num_modulators

    return y
