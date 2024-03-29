import multiprocessing
from multiprocessing import Pool
import time

def random_func(number):
    time.sleep(0.5)
    return number


if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    print("\nAvailable cores:", num_cores)
    	
    num_cores_group1 = num_cores // 2
    num_cores_group2 = num_cores - num_cores_group1

    numbers = list(range(1, 50))


    start = time.time()
    results = [random_func(num) for num in numbers]
    end = time.time()

    normal = end-start

    start = time.time()
    with Pool() as pool:
        results = pool.map(random_func, numbers)
    end = time.time()

    parallel = end-start

    print("Time for normal execution:", normal)
    print("Time parallelized:", parallel)
    