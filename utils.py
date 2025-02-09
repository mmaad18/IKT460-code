import time
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split


def time_function(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    print(f"{func.__name__} took {end - start} seconds")
    return result


def time_function_out(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    diff = end - start
    print(f"{func.__name__} took {diff} seconds")
    return result, diff


def print_time(start, message="Elapsed time"):
    end = time.perf_counter()
    print(f"{message}: {end - start} seconds")


def display_info(assignment_number: int):
    print("Author: Mohamed Yahya Maad")
    print("Course: IKT460")
    print(f"Assignment: {assignment_number}")


def display_info_project():
    print("Author: Mohamed Yahya Maad")
    print("Course: IKT460")
    print("Project: _")
    print("CUDA Available: " + str(torch.cuda.is_available()))
    print("GPU Name: " + str(torch.cuda.get_device_name(0)))


def load_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

