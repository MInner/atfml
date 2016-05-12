
from collections import namedtuple
import itertools
import inspect

import numpy as np

def all_methods_static(cls):
    for name, func in inspect.getmembers(cls, predicate=inspect.ismethod):
        setattr(cls, name, staticmethod(cls.__dict__[name]))
    return cls

def record(name, d):
    return namedtuple(name, d.keys())(**d)    

def is_record(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)

def to_record(x):
    if is_record(x):
        return x
    elif isinstance(x, dict):
        return record('Data', x)
    else:
        raise ValueError("Input Data must be either a dict() or"
                         " a record(), not %s" % type(x))

from math import ceil
@all_methods_static
class inits:
    def gaussian_init_with(*, mu=0, std=1): # named only to not forget init params
        return lambda *x: np.random.normal(mu, std, size=np.prod(x)).reshape(x)

    def identity_concat_init(n, m):
        if n == m:
            return np.identity(n)
        elif n > m:
            n_repeat = int(n/m)
            n_mod = n%m
            to_stack = [np.identity(m) for i in range(n_repeat)]
            to_stack += [np.eye(n_mod, m)]
            return np.concatenate(to_stack, axis=0)
        else: # m > n
            return inits.identity_concat_init(m, n).T

    def identity_repeat_init(n, m):
        if n == m:
            return np.identity(n)
        elif m > n: # wide matrix
            n_repeat = ceil(m/n)
            n_mod = m%n
            repeated = np.repeat(np.identity(n), n_repeat, axis=1)/n_repeat
            return repeated[:n, :m]
        else: # m > n
            return inits.identity_repeat_init(m, n).T

def assert_arr_shape(shape_dict):
    str_shapes_dict = {} # global dict for named shapes
    if not all(type(x) == tuple for x in shape_dict.keys()):
        raise ValueError("Keys must be shapes (tuples), not arrays themselves")
    for A_shape, shape in shape_dict.items():
        if len(A_shape) != len(shape):
            raise AssertionError("Expected array of shape %s, got %s" % (shape, A_shape))
        for i, true_shape_i, wanted_shape_i in zip(range(len(shape)), A_shape, shape):
            if wanted_shape_i == None: # (None, ... ) => anything works
                continue 
            elif type(wanted_shape_i) is str: # ('batch',  ) => save and check consistency
                shape_name = wanted_shape_i
                if shape_name in str_shapes_dict.keys():
                    if true_shape_i != str_shapes_dict[shape_name]:
                        raise AssertionError("Error in matching named shape '%s' within"
                                             " assertion block (%s != %s)" 
                                             % (shape_name, true_shape_i, str_shapes_dict[shape_name]))
                else:
                    str_shapes_dict[shape_name] = true_shape_i
            elif true_shape_i != wanted_shape_i:
                raise AssertionError("Expected array of shape %s, got %s (%s != %s)" 
                                     % (shape, A_shape, true_shape_i, wanted_shape_i))

    return record('AssertionConstants', str_shapes_dict)

def assert_arr_shape_test():
    n, j, k, l = 10, 20, 30, 40
    a = np.random.rand(n, j, k)
    b = np.random.rand(n, k, l)

    assert_arr_shape({
        a.shape: (None, j, k),
        b.shape: (None, k, l)
    })

    assert_arr_shape({
        a.shape: ('n_batch', j, k),
        b.shape: ('n_batch', k, l)
    })

    try:
        assert_arr_shape({
            a.shape: ('n_batch', j, k),
            b.shape: ('n_batch', j, l)
        })
    except AssertionError:
        pass
    else:
        raise RuntimeError("This test should have failed")

    try:
        c = np.random.rand(n+2, k, l)
        const = assert_arr_shape({
            a.shape: ('n_batch', j, k),
            c.shape: ('n_batch', k, l)
        })
        assert(const.n_batch == n)
    except AssertionError:
        pass
    else:
        raise RuntimeError("This test should have failed")

    print("All assert_arr_shape_test test have passed!")

# same as assert_arr_shape but for theano
def extract_const_vars_from_tensors(shape_dict):
    tensor_shapes, req_shapes = zip(*shape_dict.items())
    const_dict = {}
    for tensor_shape, req_shape in zip(tensor_shapes, req_shapes):
        for tensor_shape_i, req_shape_i in zip(tensor_shape, req_shape):
            if type(req_shape_i) == str:
                const_dict[req_shape_i] = tensor_shape_i

    return record('DataShapeSymbols', const_dict)

class behaviours:
    class WieghtLogBehaviour():
        def __init__(self, name):
            self.name = name
            self.log = []

        def each_iteration(self, loss_val, theta, data, const, info):
            self.log.append(np.copy(getattr(theta, self.name)))

    class LossLogBehaviour():
        def __init__(self):
            self.log = []

        def each_iteration(self, loss_val, theta, data, const, info):
            self.log.append(loss_val)