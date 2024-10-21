import pickle

def dump_list_as_pickle(lst, filename):
  

  with open(filename, 'wb') as f:
    pickle.dump(lst, f)

def read_list_from_pickle(filename):
  
  with open(filename, 'rb') as f:
    return pickle.load(f)

# Example usage:
my_list = [1, 2, 3, 4, 5]
filename = 'my_list.pkl'
import pdb;pdb.set_trace()
# Dump the list as a pickle
dump_list_as_pickle(my_nlist, filename)








# Read the list from the pickle


loaded_list = read_list_from_pickle(filename)
print(loaded_list)  # Output: [1, 2, 3, 4, 5]