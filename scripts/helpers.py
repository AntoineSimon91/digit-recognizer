
# standrad imports
import time


def timer(method):
    """Print the duration of the decorated function"""
    def wrap(*args, **kwargs):
        start_time = time.time()
        output = method(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        duration = round(duration, 1)
        method_name = method.__name__
        print(f"[TIMER] '{method_name}' method duration: {duration} sec")
        return output
    return wrap
