
class MovingAverage(object):
    def __init__(self, size):
        self.average = 0
        self.counter = 0
        self.size = size
        self.size_inv = 1/size
        
    def next_val(self, new_val):
        
        if self.counter >= self.size:
            total_sum = self.average * (self.size - 1) + new_val
            self.average = total_sum * self.size_inv
            
        elif self.counter < self.size:
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
            use_val = 10
        elif ii == 11:
            use_val = 110
        else:
            use_val = -5
            
        print( example_avg.counter, use_val, example_avg.next_val( use_val ) )
            
            