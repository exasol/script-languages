#!/usr/bin/env python3
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import requests
from exasol_python_test_framework import udf
from exasol_python_test_framework.exatest.utils import obj_from_json_file
from requests.auth import HTTPBasicAuth


class PytorchTest(udf.TestCase):
    def setUp(self):
        self.query('create schema pytorchbasic', ignore_errors=True)

    def test_pytorch(self):
        self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON3 SCALAR SCRIPT
                test_pytorch(epochs INTEGER)
                RETURNS VARCHAR(10000) AS
                    
                import torch
                import torch.nn as nn
                import torch.optim as optim
                import numpy as np
            
                def run(ctx):
                    # Generate random data
                    np.random.seed(42)
                    x = np.random.rand(100, 1).astype(np.float32)  # Random x values
                    y = 10 * x  # Corresponding y values
                
                    # Convert numpy arrays to torch tensors
                    x_train = torch.from_numpy(x)
                    y_train = torch.from_numpy(y)
                
                    # Define a simple linear regression model
                    class LinearModel(nn.Module):
                        def __init__(self):
                            super(LinearModel, self).__init__()
                            self.linear = nn.Linear(1, 1)  # Input and output both have size 1
                
                        def forward(self, x):
                            return self.linear(x)
                
                    # Initialize the model, loss function, and optimizer
                    model = LinearModel()
                    criterion = nn.MSELoss()
                    optimizer = optim.SGD(model.parameters(), lr=0.01)
                
                    # Training loop
                    epochs = ctx.epochs
                    for epoch in range(epochs):
                        model
                    # Check accuracy
                    model.eval()
                    with torch.no_grad():
                        y_pred = model(x_train)
                        mse = criterion(y_pred, y_train)
                        return f'Mean Squared Error: {mse.item():.4f}'
                /
                '''))

        row = self.query(f"SELECT pytorchbasic.test_pytorch(1000);")[0]
        self.assertIn('Mean Squared Error', row[0])


if __name__ == '__main__':
    udf.main()
