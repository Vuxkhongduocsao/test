def is_armstrong_number(number):
    '''
    This function determines whether a number is an Armstrong number or not.

    :number int: input number to check
    :return bool: True or False
    '''
    length = len(str(number)) 
    res = number
    sum = a = 0
    
    for i in range(length):
        a = int(res / pow(10,length - i - 1))
        res = res - a * pow(10,length - i - 1)
        sum = sum + pow(a, length)
        
    if sum == number:
        return True
    else:
        return False
    