def steps(number):
    """
    Return the number of steps to reach 1 using the Collatz Conjecture.
    
    :param number (int): Positive integer
    :return int: Number of steps to reach 1
    """
    if number <= 0:
        raise ValueError("Only positive integers are allowed")

    steps = 0
    while number != 1:
        number = number // 2 if number % 2 == 0 else number * 3 + 1
        steps += 1

    return steps
