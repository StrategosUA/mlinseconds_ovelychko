# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs
from ..utils import plotter as pt

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = solution.hidden_size
        
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        # nn.init.xavier_normal_(self.linear1.weight)
        
        self.linear2 = nn.Linear(self.hidden_size, output_size)
        # nn.init.xavier_normal_(self.linear1.weight)
        
        self.batch_norm_1D = nn.BatchNorm1d(num_features=self.hidden_size, track_running_stats=False)
        
        self.activation_function1 = solution.activation_function1
        self.activation_function2 = solution.activation_function2
        
        self.loss_function = solution.loss_function
        # self.Dropout2d = torch.nn.Dropout2d(p=0.2, inplace=False)

    def forward(self, x):
        x = self.linear1(x) # input -> hidden
        x = self.activation_function1(x)
        x = self.batch_norm_1D(x)
        x = self.activation_function1(x)
        x = self.batch_norm_1D(x)
        x = self.activation_function1(x)
        x = self.batch_norm_1D(x)        
        x = self.linear2(x) # hidden -> output
        x = self.activation_function2(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        if self.loss_function == 'Square':
           return ((output-target)**2).sum()
        elif self.loss_function == 'BCELoss':
           return nn.BCELoss()(output,target)
        elif self.loss_function == 'MSELoss':
           return nn.MSELoss()(output,target)
        elif self.loss_function == 'L1Loss':
           return nn.L1Loss()(output,target)
        
    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()  

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 0.9
        # Control number of hidden neurons
        self.hidden_size = 900
        # Control choice of loss function
        self.loss_function = 'BCELoss'
        self.momentum = 0.5
        self.activation_function1 = torch.relu_
        self.activation_function2 = torch.sigmoid     
        
        # Grid search settings, see grid_search_tutorial
        # 'Square', 'BCELoss','MSELoss','L1Loss'
        self.momentum_grid = [0.5]
        self.loss_function_grid = ['BCELoss']
        self.learning_rate_grid = [0.9]  
        self.hidden_size_grid = [900]
        
        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 100 
    
    # Return trained model
    def train_model(self, train_data, train_target, context):
        # Uncommend next line to understand grid search
        # return self.grid_search_tutorial()
        # Model represent our neural network
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        # Optimizer used for training neural network
        # sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", context.step)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        while True:
            # Report step, so we know how many steps
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or correct == total:
                # print('last step:' + str(context.step))
                break
            # calculate error
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            # self.print_stats(context.step, error, correct, total)
            
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
        
        if run_grid_search == True:
        # adding steps (our main KPI) to GridSearch
            self.grid_search.add_result('step', context.step)
            if self.iter == self.iter_number-1:
                # print("[HelloXor] chose_str={}".format(self.grid_search.choice_str))
                # print("[HelloXor] iters={}".format(self.grid_search.get_results('step')))
                stats = self.grid_search.get_stats('step')
                print("lr={} STEPS: Mean={:.2f} Std={:.2f}".format(self.grid_search.choice_str,float(stats[0]), float(stats[1])))

        return model
    
    def print_stats(self, step, error, correct, total):
               
        if step % 500 == 0: 
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))
            

    def grid_search_tutorial(self):
        # During grid search every possible combination in field_grid, train_model will be called
        # iter_number times. This can be used for automatic parameters tunning.
        if self.grid_search:
            # print("[HelloXor] learning_rate={} iter={}".format(self.learning_rate, self.iter))
            self.grid_search.add_result('iter', self.iter)
            if self.iter == self.iter_number-1:
                print("[HelloXor] chose_str={}".format(self.grid_search.choice_str))
                print("[HelloXor] iters={}".format(self.grid_search.get_results('iter')))
                stats = self.grid_search.get_stats('step')
                print("[HelloXor] Mean={} Std={} ".format(stats[0], stats[1]))
        else:
            print("Enable grid search: See run_grid_search in the end of file")
            exit(0)
        
###
###
### Don't change code after this line
###
###

class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
# run_grid_search = True
if run_grid_search:
    
    # gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
    
    grid_search = gs.GridSearch()
    results_data = gs.ResultsData.get_global()
    grid_search.run(Config(), case_number=10, results_data=results_data)
    
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)