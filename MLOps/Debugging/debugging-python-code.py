"""
@author : Ajeet Kumar
@website : https://www.ajeetkbhardwaj.github.io

"""
#%% Logging 
# 1. How to use the logging to capture and store information about the potential issues in our python code ?

import logging

def mul(x: float, y: float):
    """

    """
    # Return whether an object is an instance of a class or of a subclass thereof.
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        logging.error('Input variable are not float or integer type')
    z = x * y
    return z

def reverse_mul(num_list: list):
    """

    """
    logging.info("Length of {num_list} is {list_len}".format(num_list=num_list, list_len=len(num_list)))
    rev_list = num_list.copy()
    rev_list.reverse()
    mult_list = [mul(num_list[iter],
    rev_list[iter]) for iter in range(0, len(num_list))]
    return mult_list

num_list = [1, 'no', 5, 8, 9]
print(reverse_mul(num_list))

num_list2 = [3.9, 23.4, 99, 23]
print(reverse_mul(num_list2))
# %% Debugging in software development : Let's first find the issues and error in code.
# different version of same function could result in different issues/error or wrong output
# when running the code such as mult function
 
def mult(x, y):
    z = x * y
#z = x * y #indentation issue
    #z=x ** y #unexpected answer
    return z
    #retunr z #syntax error
print(mult(4, 5))

# how to use traceback to find necessary information for debugging a piece of Python code.
def mul(x: float, y: float):
    """

    """
    z = x * y
    return z

def reverse_mul(num_list: list):
    """

    """
    rev_list = num_list.copy()
    rev_list.reverse()
    mult_list = mul(num_list,rev_list)
    return mult_list

num_list = [1, 7, 5, 8, 9]
print(reverse_mul(num_list))
# TypeError: can't multiply sequence by non-int of type 'list'
#%%
# how wrong code indentation could result in incorrect answer while the code would run without returning an error. 
# odd_counter() that calculates number of odd numbers in an input list of numbers, which is three considering the num_list.

def odd_counter(num_list: list):
  """
  """
  odd_count = 0
  for num in num_list:
    if (num % 2) == 0:
      print("{} is even".format(num))
    else:
      print("{} is even".format(num))
      odd_count += 1

  return odd_count


num_list = [1, 2, 5, 8, 9]
print(f'Total number of odd numbers in the list: {odd_counter(num_list)}')

# Now we change indentation of odd_count += 1 which results in no error but returns total number of numbers in the input num_list instead of only odd numbers.
def odd_counter(num_list: list):
  """
  """
  odd_count = 0
  for num in num_list:
    if (num % 2) == 0:
      print("{} is even".format(num))
    else:
      print("{} is even".format(num))
    odd_count += 1

  return odd_count
num_list = [1, 2, 5, 8, 9]
print(f'Total number of odd numbers in the list: {odd_counter(num_list)}')


# how a.isnull() and a.isnan() returns AttributeError.
# a.isnull() returns AttributeError as 'list' object has no attribute 'isnull'.

a = [1, 2]
a.isnull()
#%%
# a.isnan() returns AttributeError as  'float' object has no attribute 'isnan'.

import numpy as np
a = np.nan
a.isnan() 
# %%
