class Softmax(torch.nn.Module):
  def __init__(self,
               units,
               device = device, dtype = torch.float32):
    super(Softmax, self).__init__()

    self.units, self.degree = units, degree
    self.device, self.dtype = device, dtype 

  def forward(self, X):

    X = X.to(device = self.device, dtype = self.dtype)
    
    y = torch.exp(x)/torch.exp(x).sum(1).view(-1, 1)
    
    return y
