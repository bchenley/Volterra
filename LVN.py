class LVN(torch.nn.Module):
  def __init__(self,
               ## input-output
               num_inputs = 1, num_outputs = 1, dt = 1, t0 = 0.,
               ## Filterbank Layer
               num_filters = [[1]], 
               a_init = [[0.5]], a_train = [[True]],
               ## Hidden Layer               
               num_hiddens = [1], w_init = [None], w_train = [True],
               activation = ['polynomial'], 
               degree = [1], 
               c_init = [None], c_train = [True], 
               s_init = [None], s_train = [True], 
               bs_init = [None], bs_train = [True], 
               b_init = [None], b_train = [False],  
               ## Interaction Layer
               num_interaction_hiddens = None, wi_init = None, wi_train = True, 
               interaction_activation = 'polynomial', interaction_degree = 1, 
               ci_init = None, ci_train = True,  
               bi_init = None, bi_train = False,                 
               ## Output Layer               
               wo_init = None, wo_train = True,
               bo_init = None, bo_train = True,
               ## Modulation Layer
               nonstationary = False, t_domain = None,              
               # legendre
               degree_legendre = None,  
                # chebychev
               chebychev_legendre = None,  
               # hermite
               hermite_legendre = None,               
               # fourier  
               num_modulators = None,
               f_init = None, f_train = True,
               p_init = None, p_train = True,
               # sigmoid
               num_sigmoids = None,
               sm_init = None, sm_train = True,
               bsm_init = None, bsm_train = True,
               #
               wm_init = None, wm_train = True,
               bm_init = None, bm_train = True,
               cm_init = None, cm_train = True,
               device = device, dtype = torch.float32):
    
    super(LVN, self).__init__()
    
    if len(num_filters) == 1: num_filters = num_filters*num_inputs
    if len(a_init) == 1: a_init = a_init*num_inputs
    if len(a_train) == 1: a_train = a_train*num_inputs
    
    if len(w_init) == 1: w_init = w_init*num_inputs
    if len(w_train) == 1: w_train = w_train*num_inputs

    for i_input in range(num_inputs):
      if len(a_init[i_input]) != len(num_filters[i_input]): a_init[i_input] = a_init[i_input]*len(num_filters[i_input])
      if len(a_train[i_input]) != len(num_filters[i_input]): a_train[i_input] = a_train[i_input]*len(num_filters[i_input])

    if len(activation) == 1: activation = activation*num_inputs

    if len(num_hiddens) == 1: num_hiddens = num_hiddens*num_inputs
    if len(degree) == 1: degree = degree*num_inputs    
    if len(c_init) == 1: c_init = c_init*num_inputs
    if len(c_train) == 1: c_train = c_train*num_inputs

    if len(s_init) == 1: s_init = s_init*num_inputs
    if len(s_train) == 1: s_train = s_train*num_inputs

    if len(bs_init) == 1: bs_init = bs_init*num_inputs
    if len(bs_train) == 1: bs_train = bs_train*num_inputs

    if len(b_init) == 1: b_init = b_init*num_inputs
    if len(b_train) == 1: b_train = b_train*num_inputs

    output_input_dim = np.sum(num_hiddens)

    fb_layer = torch.nn.ModuleList([])
    hidden_layer = torch.nn.ModuleList([])
    
    ## Filterbank and hidden layers  
    for i_input in range(num_inputs): # for each input
      fb_layer.append(torch.nn.ModuleList([]))
      
      for i_fb in range(len(num_filters[i_input])):
          
        fb_layer[i_input].append(LaguerreFilterbank(num_filters=num_filters[i_input][i_fb], 
                                                    a_init=a_init[i_input][i_fb], a_train=a_train[i_input][i_fb], 
                                                    dt=dt, 
                                                    dtype=dtype, device=device))

      hidden_layer.append(HiddenLayer(input_dim = np.sum(num_filters[i_input]), 
                                      output_dim = num_hiddens[i_input],
                                      activation = activation[i_input],
                                      degree = degree[i_input], 
                                      w_init = w_init[i_input], w_train = w_train[i_input], 
                                      c_init = c_init[i_input], c_train = c_train[i_input], 
                                      s_init = s_init[i_input], s_train = s_train[i_input], 
                                      bs_init = bs_init[i_input], bs_train = bs_train[i_input], 
                                      b_init = b_init[i_input], b_train = b_train[i_input], 
                                      device = device, dtype = dtype))
    ##

    ## interaction layer
    if num_interaction_hiddens is not None:  
      
      interaction_layer = HiddenLayer(input_dim = np.sum(num_hiddens),
                                      output_dim = num_interaction_hiddens, 
                                      activation = interaction_activation,
                                      degree = interaction_degree, 
                                      w_init = wi_init, w_train = wi_train, 
                                      c_init = ci_init, c_train = ci_train, 
                                      b_init = bi_init, b_train = bi_train, 
                                      device = device, dtype = dtype)
      
    else:
      interaction_layer = None
    ##
    
    ## modulation layer
    if nonstationary:  
      if interaction_layer is not None:
        input_dim_modulation = num_interaction_hiddens
      else:
        input_dim_modulation = np.sum(num_hiddens)

      modulation_layer = ModulationLayer(t_domain = t_domain,
                                         input_dim = input_dim_modulation, 
                                         # legendre
                                         degree_legendre = degree_legendre,
                                         # chebychev
                                         chebychev_legendre = chebychev_legendre,
                                         # hermite
                                         hermite_legendre = hermite_legendre,               
                                         # fourier
                                         num_modulators = num_modulators,
                                         f_init = f_init, f_train = f_train,
                                         p_init = p_init, p_train = p_train,
                                         # sigmoid
                                         num_sigmoids = num_sigmoids,
                                         s_init = sm_init, s_train = sm_train,
                                         bs_init = bsm_init, bs_train = bsm_train,
                                         #
                                         w_init = wm_init, w_train = wm_train,
                                         b_init = bm_init, b_train = bm_train,
                                         # c_init = cm_init, c_train = cm_train,
                                         device = device, dtype = dtype)
    else:
      modulation_layer = None
      t_domain = None
    ##

    ## output layer
    if modulation_layer is not None:
      input_dim_output = modulation_layer.num_modulators
      bo_init = None
      bo_train = False
    elif interaction_layer is not None:
      input_dim_output = interaction_layer.output_dim    
    else:
      input_dim_output = np.sum(num_hiddens)
      bo_init = torch.zeros(size = (1, num_outputs)).to(device = device, dtype = dtype)
      
    if (num_outputs == 1): # & (modulation_layer is None):
      wo_init = torch.ones((input_dim_output, 1)).to(device = device, dtype = dtype)
      wo_train = False
    else:
      wo_init = torch.normal(size = (input_dim_output, 1), mean = 0., std = 0.001).to(device = device, dtype = dtype)      
      # wo_train = True

    output_layer = HiddenLayer(input_dim = input_dim_output, 
                               output_dim = num_outputs,
                               activation = 'linear',
                               w_init = wo_init, w_train = wo_train, 
                               b_init = bo_init, b_train = bo_train, 
                               device = device, dtype = dtype)  
    ##

    self.fb_layer = fb_layer
    self.hidden_layer = hidden_layer
    self.interaction_layer = interaction_layer
    self.t_domain = t_domain
    self.modulation_layer = modulation_layer
    self.output_layer = output_layer

    self.num_inputs, self.num_outputs = num_inputs, num_outputs
    self.dt, self.t0 = dt, t0
    self.num_filters = num_filters
    self.num_hiddens, self.degree = num_hiddens, degree
    self.num_interaction_hiddens, self.interaction_degree = num_interaction_hiddens, interaction_degree
    self.degree_legendre, self.num_modulators, self.num_sigmoids = degree_legendre, num_modulators, num_sigmoids
    
    self.loss = 0.

    self.device, self.dtype = device, dtype

  def predict_n(self, X_n, t_n):

    X_n = X_n.to(device = self.device, dtype = self.dtype) 
    t_n = t_n.to(device = self.device, dtype = self.dtype)

    if t_n == self.t0:      
      for i_input in range(self.num_inputs):
        for i_fb in range(len(self.num_filters[i_input])):      
          self.fb_layer[i_input][i_fb].v = torch.zeros((1, self.fb_layer[i_input][i_fb].num_filters)).to(device = self.device, dtype = self.dtype)

    Z_n = []
    for i_input in range(self.num_inputs):       
      V_i_n = [[]]*len(self.num_filters[i_input])
      for i_fb in range(len(self.num_filters[i_input])):            
        V_i_n[i_fb],  self.fb_layer[i_input][i_fb].v = self.fb_layer[i_input][i_fb](X = X_n[:, self.idx_x[i_input]:(self.idx_x[i_input]+1)],
                                                                                    v = self.fb_layer[i_input][i_fb].v.detach())

      V_i_n = torch.cat(V_i_n, 1)
      
      Z_n.append(self.hidden_layer[i_input](V_i_n))

    Z_n = torch.cat(Z_n, 1)  

    Z_n = self.interaction_layer(Z_n) if self.interaction_layer is not None else Z_n
  
    if self.modulation_layer is not None:
      Z_n = self.modulation_layer(Z_n, t_n)
    
    y_pred_n = self.output_layer(Z_n)

    return y_pred_n

  def forward(self, X, t, y = None, y_prev = None):

    N, P = X.shape

    y_pred = torch.zeros(size = (N, self.num_outputs)).to(device = self.device, dtype = self.dtype)

    for n in range(N):

      if y_prev is not None: X[n:(n+1), self.idx_r_x] = y_prev.detach()[:, self.idx_r_y]
      
      y_pred[n:(n+1)] = self.predict_n(X_n = X[n:(n+1)], t_n = t[n:(n+1)])

      y_prev = y[n:(n+1), :] if y is not None else y_pred[n:(n+1), :]

    return y_pred, y_prev
  
  def fit(self,
          loss_fn, opt,
          dg_train, dg_val = None,
          apply_constraints = False,
          a_min_max = [0.1, 0.9],
          w_pos = False,
          wi_pos = False,
          apply_penalties = False,
          w_reg = [0.001, 1], c_reg = [0.001, 1],               
          wi_reg = [0.001, 1], ci_reg = [0.001, 1],
          wm_reg = [0.001, 1], 
          wo_reg = [0.001, 1], 
          epochs = 20,
          report_every_k_batches = None,
          earlystopping = False, 
          param_patience = 3, param_stop_thresh = 1e-4,
          loss_patience = 3, loss_stop_thresh = 1e-4,
          save_model = False,
          save_model_every_k_epochs = 5,
          model_path = '*lvn_temp'):
    
    self.loss_fn, self.opt = loss_fn, opt

    num_batches_train = len(dg_train.dl)
    batch_size_train = len(dg_train.dl.dataset)

    num_batches_val, batch_size_val = None, None
    if dg_val is not None:
      num_batches_val = len(dg_val.dl)
      batch_size_val = len(dg_val.dl.dataset)

    num_batches_test, batch_size_test = None, None
    if dg_test is not None:
      num_batches_test = len(dg_test.dl)
      batch_size_test = len(dg_test.dl.dataset)

    self.t0 = self.generate_Xy(dg_train)[2][0].item()

    self.idx_x, self.idx_y = dg_train.idx_x, dg_train.idx_y
    self.idx_r_x, self.idx_r_y = dg_train.idx_r_x, dg_train.idx_r_y

    self.num_batches_train, self.batch_size_train = num_batches_train, batch_size_train
    self.num_batches_val, self.batch_size_val = num_batches_val, batch_size_val
    self.num_batches_test, self.batch_size_test = num_batches_test, batch_size_test

    self.input_names, self.output_names = dg_train.input_names, dg_train.output_names
    
    self.epochs = epochs
    
    if len(a_min_max) == 1: a_min_max = a_min_max*self.num_inputs
    if len(w_reg) == 1: w_reg = w_reg*self.num_inputs
    if len(c_reg) == 1: c_reg = c_reg*self.num_inputs

    for i_input in range(self.num_inputs):
      if len(a_min_max[i_input]) != len(self.num_filters[i_input]):
        a_min_max[i_input] = a_min_max[i_input]*len(self.num_filters[i_input])
  
    self.apply_constraints = apply_constraints
    self.a_min_max = a_min_max
    self.w_pos, self.wi_pos = wi_pos, wi_pos

    self.apply_penalties = apply_penalties
    self.w_reg, self.c_reg = w_reg, c_reg
    self.wi_reg, self.ci_reg = wi_reg, ci_reg
    self.wm_reg = wm_reg
    self.wo_reg = wo_reg

    self.save_model = save_model
    self.save_model_every_k_epochs = save_model_every_k_epochs
    self.model_path = model_path

    self.report_every_k_batches = report_every_k_batches
    self.earlystopping = earlystopping 
    self.param_patience, self.param_stop_thresh = param_patience, param_stop_thresh
    self.loss_patience, self.loss_stop_thresh = loss_patience, loss_stop_thresh

    self.loss_fn, self.opt = loss_fn, opt
    
    history_train = {'step': np.empty((0, 1))}
    history_train['loss'] = np.empty((0, self.num_outputs))

    history_val = None
    if dg_val is not None:
      history_val = {'step': np.empty((0, 1))}
      history_val['loss'] = np.empty((0, self.num_outputs))
    
    for name, param in lvn.named_parameters():  
      history_train[name] = np.empty((0, param.numel()))

    self.history_train, self.history_val = history_train, history_val

    self.step_train = 0    
    num_samples_val, self.step_val = None, 0
    
    for epoch in tqdm(range(self.epochs)):

      self.v_prev = torch.zeros((1, self.num_outputs)).to(device = self.device, dtype = self.dtype)

      loss_train_epoch = self.train_step(dg_train = dg_train)
      
      loss_train_str = f"Training {self.loss_fn.loss} = {loss_train_epoch}" 

      loss_val_str = "No validation data"
      if dg_val is not None:
        loss_val_epoch = self.val_step(dg_val = dg_val)
        loss_val_str = f"Validation {self.loss_fn.loss} = {loss_val_epoch}" 
        
      a_report = ', '.join([f"α-{self.input_names[i_input][0]} = {[np.round(self.fb_layer[i_input][i_fb].a.item(),4) for i_fb in range(len(self.num_filters[i_input]))]}" for i_input in range(self.num_inputs)])        
      print(f"-Epoch {epoch+1}: {loss_train_str} | {a_report} | {loss_val_str}")

      if self.earlystopping:
        if (epoch >= self.loss_patience):
          if (self.history_train['loss'][-self.loss_patience:] < self.loss_stop_thresh).all():
            print(f"Early stopping at epoch {epoch+1}. All gradients less than {self.loss_stop_thresh}")
            torch.save(self, self.model_path)  
            print(f"Model saved...") 
            return
        elif (epoch >= self.param_patience):
          if (np.abs([param.grad.data for param in self.parameters()]) < self.param_stop_thresh).all():
            print(f"Early stopping at epoch {epoch+1}. All gradients less than {self.param_stop_thresh}")
            torch.save(self, self.model_path)  
            print(f"Model saved...") 
            return
      
      if ((epoch % self.save_model_every_k_epochs == 0) | (epoch == (self.epochs-1))): # (epoch != 0) & 
        torch.save(self, self.model_path)  
        print(f"Model saved at epoch {epoch+1}")
  
  def generate_Xy(self, dg):
    
    X, y, t = None, None, None

    for batch, (X_batch, y_batch, t_batch) in enumerate(dg.dl):

      X_batch = X_batch.to(device = self.device, dtype = self.dtype)
      y_batch = y_batch.to(device = self.device, dtype = self.dtype)
      t_batch = t_batch.to(device = self.device, dtype = self.dtype)
      
      if X is None: X = torch.empty(size = (0, X_batch.shape[-1])).to(device = self.device, dtype = self.dtype)
      if y is None: y = torch.empty(size = (0, y_batch.shape[-1])).to(device = self.device, dtype = self.dtype)
      if t is None: t = torch.empty(size = (0, t_batch.shape[-1])).to(device = self.device, dtype = self.dtype)
    
      X = torch.cat((X, X_batch), axis = 0)
      y = torch.cat((y, y_batch), axis = 0)
      t = torch.cat((t, t_batch), axis = 0)
    
    return X, y, t

  def constrain(self):

    if self.a_min_max is not None:  
      for i_input in range(self.num_inputs):
        for i_fb in range(len(self.fb_layer[i_input])):
          if self.a_min_max[i_input][i_fb] is not None:
            self.fb_layer[i_input][i_fb].a.data.clamp_(self.a_min_max[i_input][i_fb][0],
                                                        self.a_min_max[i_input][i_fb][1])
        
      if self.w_pos:
        self.hidden_layer[i_input].w = self.hidden_layer[i_input].w.sum(0).sign()*self.hidden_layer[i_input].w
    
    if self.wi_pos:
      self.interaction_layer.w = self.interaction_layer.w.sum(0).sign()*self.interaction_layer.w

  def penalize(self):

    penalty = 0.

    for i_input in range(self.num_inputs):
      # hidden layer
      penalty += self.w_reg[0]*self.hidden_layer[i_input].w.norm(self.w_reg[1])*int(self.hidden_layer[i_input].w.requires_grad)
      if self.hidden_layer[i_input].activation == 'polynomial':        
        penalty += self.c_reg[0]*self.hidden_layer[i_input].F.c.norm(self.c_reg[1])*int(self.hidden_layer[i_input].F.c.requires_grad)

    # interaction layer
    if self.interaction_layer is not None:
      penalty += self.wi_reg[0]*self.interaction_layer.w.norm(self.wi_reg[1])*int(self.interaction_layer.w.requires_grad)
      if self.interaction_layer.activation == 'polynomial':
        penalty += self.ci_reg[0]*self.interaction_layer.F.c.norm(self.ci_reg[1])*int(self.interaction_layer.F.c.requires_grad)

    # output layer
    penalty += self.wo_reg[0]*self.output_layer.w.norm(self.wo_reg[1])*int(self.output_layer.w.requires_grad)

    # modulation layer
    if self.modulation_layer is not None:
      penalty += self.wm_reg[0]*self.modulation_layer.w.norm(self.wm_reg[1])
      
    return penalty

  def train_step(self, dg_train):
    
    num_batches = len(dg_train.dl)
    num_samples = len(dg_train.dl.dataset)

    loss_train = 0.
    current_batch_size = 0

    # self.v_prev = 0. # dg_train.y_prev
    
    self.train()
    for batch, (X_batch, y_batch, t_batch) in enumerate(dg_train.dl):
        
        X_train_batch = X_batch.to(device = self.device, dtype = self.dtype)
        y_train_batch = y_batch.to(device = self.device, dtype = self.dtype)
        t_train_batch = t_batch.to(device = self.device, dtype = self.dtype)

        y_train_pred_batch, self.v_prev  = self.forward(X = X_train_batch, t = t_train_batch, y = y_train_batch, y_prev = self.v_prev)
        
        # calculate and accumalate loss      
        loss_train_batch = self.loss_fn(y_train_pred_batch, y_train_batch)
        
        # calcluate penalties
        if self.apply_penalties: loss_train_batch += self.penalize()
        
        # Backpropagation
        self.opt.zero_grad()   
        
        # add a_penalty to loss if desired
        loss_train_batch.backward()
        
        # make one update to model parameters                  
        self.opt.step()

        # contraints
        if self.apply_constraints: self.constrain()

        current_batch_size += X_train_batch.shape[0]        
        loss_train += loss_train_batch.cpu().detach().numpy()        
        
        self.history_train['step'] = np.concatenate((self.history_train['step'], np.reshape(int(self.step_train),(1, 1))), axis = 0)
        self.history_train['loss'] = np.concatenate((self.history_train['loss'], np.reshape(loss_train_batch.detach().cpu(),(1, self.num_outputs))), axis = 0)

        for name, param in self.named_parameters(): 
          self.history_train[name] = np.concatenate((self.history_train[name], param.cpu().detach().numpy().reshape(1, -1)), axis = 0)

        if self.report_every_k_batches is not None:
          if ((batch % self.report_every_k_batches == 0) | (batch == (num_batches-1))):
            print(f"Batch {batch+1} Training Loss = {loss_train_batch.detach().cpu().numpy().round(4)} {current_batch_size}/{num_samples}") 

        self.step_train += 1

    loss_train /= num_batches

    return loss_train

  def val_step(self, dg_val):

    num_batches = len(dg_val.dl)
    num_samples = len(dg_val.dl.dataset)
    
    loss_val = 0.

    # y_prev_val = dg_val.y_prev

    self.eval()
    for batch, (X_batch, y_batch, t_batch) in enumerate(dg_val.dl):
        
      X_val_batch = X_batch.to(device = self.device, dtype = self.dtype)
      y_val_batch = y_batch.to(device = self.device, dtype = self.dtype)
      t_val_batch = t_batch.to(device = self.device, dtype = self.dtype)
      
      y_val_pred_batch, self.v_prev = self.forward(X = X_val_batch, t = t_val_batch, y = y_val_batch, y_prev = self.v_prev)
      
      # calculate and accumalate loss
      loss_val_batch = self.loss_fn(y_val_pred_batch, y_val_batch)  
      
      loss_val += loss_val_batch.cpu().detach().numpy()
      
      self.history_val['step'] = np.concatenate((self.history_val['step'], np.reshape(int(self.step_val),(1, 1))), axis = 0)
      self.history_val['loss'] = np.concatenate((self.history_val['loss'], np.reshape(loss_val_batch.detach().cpu().numpy().round(4),(1, self.num_outputs))), axis = 0)

      # self.step_val += 1

    loss_val /= num_batches

    return loss_val

class DataGenerator():
  def __init__(self,
               df,
               input_names, output_names, time_name,
               batch_size = 32):

    # collect input/output names
    data_names = {name: i for i,name in enumerate(df.columns)}
    self.data_names = data_names
    self.input_names = input_names
    self.output_names = output_names
    self.time_name = time_name

    # collect data
    data =  torch.tensor(df.values)

    _, idx_x, _ = np.intersect1d(input_names, df.columns, return_indices = True)
    _, idx_y, _ = np.intersect1d(output_names, df.columns, return_indices = True)
    _, idx_r_x, idx_r_y = np.intersect1d(input_names, output_names, return_indices = True)

    if len(idx_r_x) > 0:
      y_prev = data[0:1, [data_names[name] for name in np.array(output_names)]] # [idx_r_y]
    else:
      y_prev = None
    
    self.y_prev = y_prev
    self.data = data
    self.batch_size = batch_size

    self.idx_x, self.idx_y = idx_x, idx_y
    self.idx_r_x, self.idx_r_y = idx_r_x, idx_r_y

    # get dataloader
    self.dl = self.dataloader_
  
  @property
  def dataloader_(self):
    return self.make_dataloader(self.data) 

  class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                  data,
                  data_names, input_names, output_names, time_name):

      self.data = data
      self.len, _ = data.shape
      self.data_names, self.input_names, self.output_names, self.time_name = data_names, input_names, output_names, time_name
      self.num_inputs, self.num_outputs = len(self.input_names), len(self.output_names)
      
      self.num_samples = self.len - int(any([name in input_names for name in output_names]))

      self.samples, self.labels, self.times = self.get_samples()

    def get_samples(self):
      
      X = torch.zeros((self.num_samples, self.num_inputs))
      y = torch.zeros((self.num_samples, self.num_outputs))
      t = torch.zeros((self.num_samples, 1))

      for sample in range(self.num_samples):

        for i_x, x_name in enumerate(self.input_names):

          if x_name not in self.output_names:
            X[sample, i_x] = self.data[sample, [self.data_names.get(key) for key in [x_name]]]

        for i_y, y_name in enumerate(self.output_names):

          y[sample, i_y] = self.data[sample, [self.data_names.get(key) for key in [y_name]]]

        t[sample] = self.data[sample, [self.data_names.get(key) for key in [self.time_name]]]

      return X, y, t
  
    def __len__(self): 
      return self.num_samples

    def __getitem__(self, idx):
      return self.samples[idx], self.labels[idx], self.times[idx]

  def make_dataloader(self, data):

    ds = self.Dataset(data, 
                      data_names = self.data_names, 
                      input_names = self.input_names, output_names = self.output_names,
                      time_name = self.time_name)

    return torch.utils.data.DataLoader(ds,
                                      batch_size = self.batch_size, 
                                      shuffle = False)
 
