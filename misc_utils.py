import multiprocessing

import psutil


def available_memory_gb():
    available_memory = psutil.virtual_memory().available
    return available_memory / (1024 ** 3)

def used_memory_gb():
    used_memory = psutil.virtual_memory().used
    return used_memory / (1024 ** 3)

def total_memory_gb():
    total_memory = psutil.virtual_memory().total
    return total_memory / (1024 ** 3)

num_workers = multiprocessing.cpu_count()  # Use all available CPU cores
def MULTIPROCESS_loop(loop_func, data):
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(loop_func, data)

    return results

########### Following for TESTing purpose ##########
####################################################

# Define your function to process each (emotion, melody) pair
def process_melody(data):
    emotion, melody = data
    # Perform processing here
    result = f"Processed {emotion} with melody {melody}"  # Replace with actual processing
    return result

if __name__ == "__main__":
    # Your list of (emotion, melody) pairs
    melodies_with_emotions = [("Q1", "melody1"), ("Q2", "melody2"), ("Q3", "melody3")]
    print(MULTIPROCESS_loop(process_melody, melodies_with_emotions))
