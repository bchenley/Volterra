class Loss():
  def __init__(self, loss = 'mse'):
    self.loss = loss

  def __call__(self, y_pred, y_true):
    if self.loss == 'mse':
      loss = torch.nn.MSELoss()(y_pred, y_true)
    elif self.loss == 'nmse':
      loss = torch.nn.MSELoss()(y_pred, y_true) / torch.pow(y_true - y_true.mean(0),2).mean()
    
    return loss
