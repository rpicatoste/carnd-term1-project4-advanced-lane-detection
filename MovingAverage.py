import numpy as np


class MovingAverage(object):
    def __init__(self, window_size):
        self.average = 0
        self.counter = 0
        self.window_size = window_size
        self.window_size_inv = 1/window_size
        
    def next_val(self, new_val):
        
        if new_val == None:
            return self.average
        
        if self.counter >= self.window_size:
            total_sum = self.average * (self.window_size - 1) + new_val
            self.average = total_sum * self.window_size_inv
            
        elif self.counter < self.window_size:
            total_sum = self.average * (self.counter) + new_val
            self.counter += 1
            self.average = total_sum / self.counter
            
        else:
            self.average = new_val
            self.counter += 1
        
        return self.average 
        
    def restart(self):
        self.average = 0
        self.counter = 0
        
    
if __name__ == '__main__':
    
    example_avg = MovingAverage(10)
    
    for ii in range(50):
        if ii < 11:
            use_val = np.array([10,20])
        elif ii == 11:
            use_val = np.array([110,220])
        else:
            use_val = np.array([-50,300])
            
        print( example_avg.counter, use_val, example_avg.next_val( use_val ) )
            
            