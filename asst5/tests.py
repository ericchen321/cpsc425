# tests for various functions in this assignment

import numpy as np
from util import load, build_vocabulary, get_bags_of_sifts, convert_label_to_integer, convert_label_to_one_hot, convert_int_to_classname

# tests for convert_label_to_integer
label_one_hot_1 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
label_int_1 = np.array([0, 2, 1])
assert np.linalg.norm(convert_label_to_integer(label_one_hot_1) - label_int_1) < 1e-9

# tests for convert_label_to_one_hot
assert np.linalg.norm(convert_label_to_one_hot(label_int_1, 3) - label_one_hot_1) < 1e-9

# tests for convert_int_to_classname
classnames = np.asarray(['cat', 'dog', 'eel'])
assert np.array_equal(convert_int_to_classname(label_int_1, classnames), np.asarray(['cat', 'eel', 'dog']))