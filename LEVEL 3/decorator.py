import time

def time_decorator(func):
    def wrapper(*args,**kwargs):

        start_time=time.time()

        result=func(*args,**kwargs)

        elapped_time=time.time()-start_time

        print(f"Elapsed time: {elapped_time:.2f} seconds")
        
        return result
    return wrapper


def sample_function():
    time.sleep(1) 

sample_function()



