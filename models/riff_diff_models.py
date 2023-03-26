import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

## Torch Essentials ####################
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def dataloader_from_dataset(dataset, batch_size: int, val_fraction:float=0.3, seed=None, device=None) -> tuple:
    '''
    '''
    # check for device
    device = device or get_default_device()

    # define sizes of training and validation set
    ds_size = len(dataset)
    val_size = int(round(len(dataset)*val_fraction, 0))
    train_size = ds_size - val_size
    
    # split
    if seed: train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    else: train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Size of Training and Validation Set: {train_size}, {val_size}")
    
    # create dataloader for CPU
    train_loader_cpu = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_cpu = DataLoader(val_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # if possible create DataLoader for GPU
    print(f"Loading Data to device: {device}")
    train_loader = DeviceDataLoader(train_loader_cpu, device)
    val_loader = DeviceDataLoader(val_loader_cpu, device)

    return train_loader, val_loader

def testset_dataloader_from_dataset(dataset, batch_size: int, device=None):
    '''
    '''
    device = device or get_default_device()

    # dataloader for cpu:
    ts_loader_cpu = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # if possible, create DataLoader for GPU
    print(f"Loading Data to device: {device}")
    ts_dataloader = DeviceDataLoader(ts_loader_cpu, device)

    return ts_dataloader

def parse_dataset(dataset):
    '''
    '''
    outdict = dict()

    a, b, c = np.split(dataset, [9,12], axis=1)
    outdict["o3"] = [[torch.from_numpy(x), torch.from_numpy(y)] for x, y in zip(a, b)]
    outdict["o1"] = [[torch.from_numpy(x), torch.from_numpy(y)] for x, y in zip(a, c)]

    a, b, c = np.split(dataset, [9,11], axis=1)
    outdict["o2"] = [[torch.from_numpy(x), torch.from_numpy(y)] for x, y in zip(a, b)]

    a, b, c, d, _ = np.split(dataset, [9,10,11,12], axis=1)
    outdict["plddt"] = [[torch.from_numpy(x), torch.from_numpy(y)] for x, y in zip(a, b)]
    outdict["rmsd"] = [[torch.from_numpy(x), torch.from_numpy(y)] for x, y in zip(a, c)]
    outdict["chainbreak"] = [[torch.from_numpy(x), torch.from_numpy(y)] for x, y in zip(a, d)]

    return outdict

# Models ############################

class Model_h1(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size, dtype=torch.float64)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size,  dtype=torch.float64)

    def forward(self, xb):
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out

    def training_step(self, batch):
        inputs, labels = batch
        out = self(inputs)                  # Generate predictions
        loss = F.mse_loss(out, labels)      # Calculate loss
        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        out = self(inputs)                    # Generate predictions
        loss = F.mse_loss(out, labels)        # Calculate loss
        acc = accuracy(out, labels)         # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def predict_batch(self, batch) -> tuple:
        """Returns (output, labels)"""
        inputs, labels = batch
        out = self(inputs)
        return out, labels

    def epoch_end(self, epoch, result):
        '''
        Keeps track of training
        '''
        results_strings = [f"{k}: {v:.4f}" for k, v in result.items()]
        if epoch % 10 == 0: print(f"Epoch [{epoch}], {chr(9).join(results_strings)}")
### MODEL FNN_BN

class Model_FNN_BN(nn.Module):
    """Feedfoward neural network with x hidden layers"""
    def __init__(self, layers: list[int], device=None, act_function=F.relu, dropout=0.1):
        super().__init__()
        # construct layers:
        self.device = device or get_default_device()
        self.layers, self.bn_layers = self.construct_hidden_layers(layers)
        self.activation = act_function
        self.dropout = nn.Dropout(dropout)
        
    def construct_hidden_layers(self, layers_list: list[int]) -> list[nn.Linear]:
        '''
        '''
        def create_size_tuples(sizes_list: list[int]) -> list[tuple]:
            return [(size, sizes_list[i+1]) for i, size in enumerate(sizes_list[:-1])]
        
        # construct layer in- and output sizes from the list of layers
        size_tuples = create_size_tuples(layers_list)
        nn_layers = list()
        bn_layers = list()
        
        # construct layers and set them as attributes of the Module, also create attribute that contains all layers at once (self.layers)
        for i, size_tuple in enumerate(size_tuples):
            in_size, out_size = size_tuple
            layer = nn.Linear(in_size, out_size, dtype=torch.float64)
            bn_layer = nn.BatchNorm1d(out_size, device=self.device, dtype=torch.float64)
            setattr(self, f"linear_{str(i).zfill(4)}", layer)
            setattr(self, f"bn_{str(i).zfill(4)}", bn_layer)
            nn_layers.append(layer)
            bn_layers.append(bn_layer)
        
        return nn_layers, bn_layers
        
    def forward(self, xb):
        layer_input = xb
        
        # run inputs through layers (except last one, where no activation function is applied):
        for layer, bn_layer in zip(self.layers[:-1], self.bn_layers[:-1]):
            # run input through layer
            batch_norm_input = layer(layer_input)
            
            # normalize
            activation_F_input = bn_layer(batch_norm_input)
            
            # run layer output through activation Function
            layer_input = self.activation(self.dropout(activation_F_input))
            
        # Get predictions using last layer
        out = self.layers[-1](layer_input)
        return out
    
    def training_step(self, batch):
        inputs, labels = batch
        out = self(inputs)                  # Generate predictions
        loss = F.l1_loss(out, labels)      # Calculate loss
        return loss
    
    def validation_step(self, batch):
        self.eval()
        inputs, labels = batch 
        out = self(inputs)                    # Generate predictions
        loss = F.l1_loss(out, labels)        # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        self.train()                        
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def predict_batch(self, batch) -> tuple:
        """Returns (output, labels)"""
        inputs, labels = batch
        out = self(inputs)
        return out, labels
    
    def epoch_end(self, epoch, result):
        '''
        Keeps track of training
        '''
        results_strings = [f"{k}: {v:.4f}" for k, v in result.items()]
        if epoch % 10 == 0: print(f"Epoch [{epoch}], {chr(9).join(results_strings)}")

###################
        
def accuracy(outputs, labels):
    loss = torch.mean((outputs - labels) ** 2)
    return loss

def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Calculates the Pearson correlation coefficient between two 1-dimensional
    PyTorch tensors.

    Args:
    - x: 1-dimensional PyTorch tensor
    - y: 1-dimensional PyTorch tensor

    Returns:
    - Pearson correlation coefficient between x and y
    """
    assert x.ndimension() == 1 and y.ndimension() == 1, f"Input tensors must be 1-dimensional, dimensions: x: {x.ndimension()} y: {y.ndimension()}"
    assert x.shape == y.shape, f"Input tensors must have the same shape"

    x_mean = x.mean()
    y_mean = y.mean()
    x_var = x.var()
    y_var = y.var()
    covariance = ((x - x_mean) * (y - y_mean)).mean()
    return covariance / (torch.sqrt(x_var) * torch.sqrt(y_var))

def pearson_r_by_column(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Pearson correlation coefficient between each column of
    outputs and labels.

    Args:
    - outputs: 2-dimensional PyTorch tensor of shape (batch_size, columns)
    - labels: 2-dimensional PyTorch tensor of shape (batch_size, columns)

    Returns:
    - 1-dimensional PyTorch tensor of Pearson correlation coefficients, one for each column
    """
    assert outputs.shape == labels.shape, "Input tensors must have the same shape"
    #assert outputs.ndimension() == 2, "Input tensors must be 2-dimensional"

    num_columns = outputs.shape[1]
    r_values = torch.zeros(num_columns, dtype=torch.float64)
    for i in range(num_columns):
        r_values[i] = pearson_r(outputs[:, i], labels[:, i])
    return r_values

class Model_h2(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size: int, hidden1_size: int, hidden2_size: int, out_size: int):
        super().__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(in_size, hidden1_size, dtype=torch.float64)
        # hidden layer 2
        self.linear2 = nn.Linear(hidden1_size, hidden2_size, dtype=torch.float64)
        # output layer
        self.linear3 = nn.Linear(hidden2_size, out_size,  dtype=torch.float64)

    def forward(self, xb):
        # Get intermediate outputs using hidden layer 1
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get intermediate outputs using hidden layer 2
        out = self.linear2(out)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using layer 3
        out = self.linear3(out)
        return out

    def training_step(self, batch):
        inputs, labels = batch
        out = self(inputs)                  # Generate predictions
        loss = F.mse_loss(out, labels)      # Calculate loss
        return loss

    def validation_step(self, batch):
        self.eval()
        inputs, labels = batch
        out = self(inputs)                    # Generate predictions
        loss = F.mse_loss(out, labels)        # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        self.train()
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def predict_batch(self, batch) -> tuple:
        """Returns (output, labels)"""
        inputs, labels = batch
        out = self(inputs)
        return out, labels

    def epoch_end(self, epoch, result):
        '''
        Keeps track of training
        '''
        results_strings = [f"{k}: {v:.4f}" for k, v in result.items()]
        if epoch % 10 == 0: print(f"Epoch [{epoch}], {chr(9).join(results_strings)}")
        #print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

## TRAINING ##########################

def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs) # returns dictionary with 'val_loss' and 'val_accuracy' of epoch

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, momentum=0.9, weight_decay=0):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    for epoch in range(epochs):
        # Training Phase 
        losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result["training_loss"] = np.mean(losses)
        model.epoch_end(epoch, result) # prints out training progress
        history.append(result)
    return history

def fit_adam(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam, betas=(0.9, 0.999)):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr, betas=betas)
    for epoch in range(epochs):
        # Training Phase 
        losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result["training_loss"] = np.mean(losses)
        model.epoch_end(epoch, result) # prints out training progress
        history.append(result)
        optimizer.zero_grad()
        if is_overfitting(history): return history
    return history

def is_overfitting(history, overfit_threshold:float=0.03, lookback:int=12, waiting_period:int=40) -> bool:
    '''
    Check if a model is overfitting based on its history of training and validation losses.
    
    Parameters:
    history (list of dicts): A list of dictionaries containing the training and validation loss of a model during training.
    overfit_threshold (float, optional): The threshold for the difference between the minimum validation loss and the mean validation loss within the lookback window. Default is 0.03.
    lookback (int, optional): The number of most recent validation losses to use for determining overfitting. Default is 12.
    waiting_period (int, optional): The minimum number of training iterations required before overfitting can be checked. Default is 40.
    
    Returns:
    bool: True if the model is overfitting, False otherwise. If the number of training iterations is less than the waiting_period, False is returned.
    '''
    if len(history) < waiting_period: return False
    val_loss_list = [x["val_loss"] for x in history]
    lookback_loss_list = val_loss_list[-1*lookback:]
    min_val_loss = np.min(val_loss_list)
    lookback_mean = np.mean(lookback_loss_list)
    
    # check if minimum loss was not within the lookback interval, if yes, return False:
    if min_val_loss == np.min(lookback_loss_list[-1*int(lookback/1.5):]): return False
    elif min_val_loss > (lookback_mean - min_val_loss*overfit_threshold): return False
    else: return True

def run_testset(model, testset_dataloader):
    '''
    '''
    outputs_list = []
    labels_list = []
    model.eval()
    for batch in testset_dataloader:
        with torch.no_grad():
            outputs, labels = model.predict_batch(batch)
        outputs_list.append(outputs)
        labels_list.append(labels)
    model.train()
    return outputs_list, labels_list

def concatenate_tensors_numpy(tensor_list: list) -> np.array:
    '''
    '''
    return np.concatenate([x.cpu().detach().numpy() for x in tensor_list])

def calculate_columnwise_correlations(data1, data2):
    n = len(data1[0])
    corrs = []
    for i in range(n):
        column1 = [x[i] for x in data1]
        column2 = [x[i] for x in data2]
        corr = np.corrcoef(column1, column2)[0,1]
        corrs.append(corr)
    return corrs

## TESTING ############################################

def test_model_on_testset(model, testset):
    '''
    '''
    # Create DataLoader of testset:
    testset_dataloader = testset_dataloader_from_dataset(testset, batch_size=len(testset))
    outputs, labels = run_testset(model, testset_dataloader)
    
    outputs = concatenate_tensors_numpy(outputs)
    labels = concatenate_tensors_numpy(labels)
    
    corrs = calculate_columnwise_correlations(outputs, labels)
    return corrs

## UTILS ################################################
def violinplot_dataset(input_data) -> np.array:
    '''
    '''
    plt.violinplot((s := [[x[i] for x in input_data] for i in range(input_data.shape[1])]))
    plt.show()
    return s

## DATA PREP #############################################
def calculate_scaling_parameters(input_data: np.array) -> dict:
    """
    Calculates the mean and variance of each column in the input data.
    :param input_data: A numpy array of data.
    :return: A dictionary containing the scaling parameters for each column.
    """
    columns = range(input_data.shape[1])
    mean_values = np.mean(input_data, axis=0)
    var_values = np.std(input_data, axis=0)
    return {column: (mean, var) for column, mean, var in zip(columns, mean_values, var_values)}

def scale_dataset(input_data: np.array, scaling_parameters: dict) -> np.array:
    """
    Scales each column in input_data to unit variance using the scaling_parameters dictionary.
    :param input_data: A numpy array of data.
    :param scaling_parameters: A dictionary containing the scaling parameters for each column.
    :return: A numpy array of scaled data.
    """
    assert len(scaling_parameters.keys()) == input_data.shape[1] #: "Scaling Params must have same number of indeces than input_data has columns!"
    scaled_data = np.copy(input_data)
    for i, (mean, var) in scaling_parameters.items():
        scaled_data[:, i] = (scaled_data[:, i] - mean) / (np.sqrt(var)*2)
    return scaled_data

def remove_rows_by_value(dataset, i, threshold, operator="<="):
    '''
    '''
    if operator == "<=": mask = dataset[:,i] <= threshold
    elif operator == ">=": mask = dataset[:,i] >= threshold
    elif operator == "<": mask = dataset[:,i] < threshold
    elif operator == ">": mask = dataset[:,i] > threshold
    else: raise ValueError(f"Operator {operator} not supported in this function.")
    
    data = dataset[mask]
    print(f"Removed {dataset.shape[0] - data.shape[0]} rows from input data, where values in col {str(i)} were {operator} {threshold}")
    return data
