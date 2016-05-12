
import abc
import collections
from types import SimpleNamespace
import numpy as np

from atfml.utils import *
class AbstractModelLoss(object):    
    def __init__(self, *, data_template, weight_template, default_init_method, 
                 optimization_method, default_type='float64', allow_downcast=False, 
                 test_forward_run=True, verbose=True, behaviours = {}, weight_bundles=None):
        
        if allow_downcast:
            raise NotImplementedError("allow_downcast does not work properly with theano")
        
        self._process_wieght_bundles(weight_bundles, weight_template)
        
        self.default_dtype = 'float64'
        self.args = SimpleNamespace()
        self.args.__dict__.update({
            'test_forward_run': test_forward_run,
            'verbose': verbose,
            'default_init_method': default_init_method,
            'weight_template': weight_template,
            'data_template': data_template,
            'optimization_method': optimization_method,
            'allow_downcast': allow_downcast,
            'behaviours': behaviours
        })
        
        self._parse_data_shapes()
        self._parse_weight_shapes()
        self._parse_optimization_method()
        
        self._collect_exported_functions()
        
    def _process_wieght_bundles(self, weight_bundles, weight_template):
        self.bundles = SimpleNamespace()
        if weight_bundles != None:
            for bundle_name, bundle_dict in weight_bundles.items():
                bundle_class = bundle_dict['class']
                bundle_obj = bundle_class(bk=self.bk, name=bundle_name, **bundle_dict['args'])
                setattr(self.bundles, bundle_name, bundle_obj)
                weight_template.update(bundle_obj.weight_template)
                bundle_obj.registered = True
        
    def _parse_data_shapes(self):
        req_names, spec_dicts = zip(*self.args.data_template.items())
        req_shapes = [spec_dict['shape'] for spec_dict in spec_dicts]
        req_dtypes = [spec_dict.get('dtype', self.default_dtype) for spec_dict in spec_dicts]
        self.d_names, self.d_shapes, self.d_dtypes = req_names, req_shapes, req_dtypes
    
    def _parse_weight_shapes(self):
        self.w_names, w_spec_dicts = zip(*self.args.weight_template.items())
        default_init = self.args.default_init_method
        self.w_shapes, self.w_inits = zip(*[(w_dict['shape'], 
                                             w_dict.get('init_method', default_init))
                                             for w_dict in w_spec_dicts])
        
        sizes = [np.prod(wi) for wi in self.w_shapes]
        if self.args.verbose:
            print("Weight shapes are: {}, n_total_params: {}".format(dict(zip(self.w_names, self.w_shapes)), 
                                                                     sum(sizes)))
        
    def _parse_optimization_method(self):
        opt_dict = self.args.optimization_method.copy() # dict here
        name = opt_dict['name']
        opt_dict.pop('name')
        params = opt_dict
        self.optimization_method = self._optimization_method_builder(name, params)
        
    def _test_data_shape(self, data_nt):
        data_pieces = [getattr(data_nt, name) for name in self.d_names]
        true_shapes = [data_i.shape for data_i in data_pieces]
        true_dtypes = [data_i.dtype for data_i in data_pieces]
        
        data_downcasted = []
        for true_dt, req_dt, name_i, data_i in zip(true_dtypes, self.d_dtypes, self.d_names, data_pieces):
            if true_dt == req_dt:
                data_downcasted.append(data_i)
            else: # types does not fit
                if self.args.allow_downcast:
                    if str(req_dt)[:3] == str(req_dt)[:3]: # int8 ~= int32, float32 ~= float64
                        if self.args.allow_downcast != 'silent':
                            print("Trying to downcast %s %s->%s, you'd better fix it!" % (name_i, true_dt, req_dt))
                        data_downcasted.append(data_i.astype(req_dt))
                        continue
                raise AssertionError("Not all dtypes match, expected %s got %s" % (self.d_dtypes, true_dtypes))
                
        const = assert_arr_shape(dict(zip(true_shapes, self.d_shapes)))
        return const, record('Data', dict(zip(self.d_names, data_downcasted)))
    
    def _collect_exported_functions(self):
        self._exported_functions = []
        for key, val in self.__class__.__dict__.items():
            if hasattr(val, 'compilation_required') and getattr(val, 'compilation_required'):
                self._exported_functions.append((key,val))
                
        self._exported_functions.append(('loss', self.__class__.loss)) # we always need to compile loss
        
    @abc.abstractmethod
    def loss(self, theta, data, const):
        pass
            
    @abc.abstractmethod
    def step_callback(self, loss, theta, data, const, info):
        # must return False when finished optimizing
        pass
    
    # implemented in backend Loss
    @abc.abstractmethod
    def fit(self, data, n_max_steps=100):
        pass
    
    # inplemented in backend Loss
    @abc.abstractmethod
    def _optimization_method_builder(self, name, **kwargs):
        pass
    
    # inplemented in backend Loss
    @abc.abstractmethod
    def _compile_and_disable_exported_functions(self):
        pass
class AutogradModelLoss(AbstractModelLoss):
    import autograd
    import autograd.numpy as anp
    import climin
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compile_and_disable_exported_functions()
        self._build()
    
    def fit(self, data, n_max_steps=100):
        x, theta, loss1d = self.__build_packed_loss()
        loss1d_grad = self.autograd.grad(loss1d)
        self.__test_run(loss1d, loss1d_grad, x, data)
        data_iterator = self.__warp_data_into_iterator(data)
        
        opt = self.optimization_method(x, loss1d_grad, data_iterator)
        with np.errstate(all='raise'): # raise errors on number overflows
            for info in opt:
                data_i = info['kwargs']['data']
                const_i, data_i = self._test_data_shape(data_i)
                our_info_i = {'n_iter': info['n_iter']}
                loss_val_i = self.compiled.loss(theta, data_i, const_i)
                ret = self.step_callback(loss_val_i, theta, data_i, const_i, our_info_i)
                for behaviour in self.args.behaviours.values():
                    behaviour.each_iteration(loss_val_i, theta, data_i, const_i, our_info_i)
                if ret == False or info['n_iter'] == n_max_steps:
                    break
                    
        self.behaviours = record('Behaviours', self.args.behaviours)
        return theta
    
    def _build(self):
        pass
    
    def _optimization_method_builder(self, name, opt_params):
        clip_val = opt_params.pop('clip', None)
        def clipped(grad):
            if clip_val == None:
                return grad
            
            def func(*argv, **kwargs):
                g_val = grad(*argv, **kwargs)
                g_clipped_val = self.anp.clip(g_val, -clip_val, clip_val)
                return g_clipped_val
            return func
        
        # external API name: internal name
        kw_universal_mapping = {'learning_rate': 'step_rate', 'clip': '_'}
        our_kwargs = {kw_universal_mapping[name]:value for name, value in opt_params.items()}
        methods = {
            'adam': lambda x, g, dat_it: self.climin.adam.Adam(x, clipped(g), args=dat_it, **our_kwargs)
        }
        
        return methods[name]
    
    def __warp_data_into_iterator(self, data):
        needs_warp = isinstance(data, dict) or not isinstance(data, collections.abc.Iterable)
        data_seq = [data] if needs_warp else data
        return itertools.cycle([([], {'data': to_record(data_i)}) for data_i in data_seq])
    
    def __build_packed_loss(self):
        x_init, theta_init = self.__build_initial_x_theta()
        def packed_loss(x, data):
            const, data = self._test_data_shape(data)
            theta = self.__unpack_x(x)
            return self.raw.loss(theta, data, const) # need to compile loss before this line
        return x_init, theta_init, packed_loss
    
    def __build_initial_x_theta(self):
        x_init = self.anp.concatenate([self.anp.ravel(init_method(*shape)) 
                                 for shape, init_method 
                                 in zip(self.w_shapes, self.w_inits)])
        sizes = [self.anp.prod(wi) for wi in self.w_shapes]
        idx = self.anp.cumsum(sizes)
        d = [(name, sub_x.reshape(shp)) for sub_x, name, shp 
             in zip(self.anp.split(x_init, idx), self.w_names, self.w_shapes)]
        theta_init = record('ModelParameters', dict(d))
        return x_init, theta_init
    
    def __unpack_x(self, x):
        sizes = [self.anp.prod(wi) for wi in self.w_shapes]
        idx = self.anp.cumsum(sizes)
        d = [(name, sub_x.reshape(shp)) for sub_x, name, shp 
             in zip(self.anp.split(x, idx), self.w_names, self.w_shapes)]
        theta = record('ModelParameters', dict(d))
        return theta
    
    def __test_run(self, loss1d, loss1d_grad, x, data):
        tmp_data_interator = self.__warp_data_into_iterator(data)
        data_clim_pack = next(tmp_data_interator)
        if self.args.test_forward_run:
            try:
                loss1d(x, **data_clim_pack[1])
            except:
                print("Error while testing forward pass!")
                raise

            try:
                loss1d_grad(x, **data_clim_pack[1])
            except:
                print("Error while testing backward pass!")
                raise
                
    def _compile_and_disable_exported_functions(self):
        import copy
        self.compiled = SimpleNamespace()
        self.raw = SimpleNamespace()
        
        def error_stub(for_name):
            def stub(*argv, **kwargs):
                raise RuntimeError("You shouldn't use model.{0} in user code"
                                   ", use model.compiled.{0} for execution (in step_callback);"
                                   " and self.raw.{0} for symbolic function (in loss\other"
                                   " compiled funcs)".format(for_name))
            return stub
        
        def self_func_stub(func):
            # stub lives in namespace and does not get self
            # but the original method self.func needs self, 
            # but it has it binded already 
            def stub(*argv, **kwargs):
                return func(self, *argv, **kwargs)
            return stub
        
        ## in case of autograd we don't do anything actually
        ## we just forward same calls into self.compiled.func_name 
        for func_name, func in self._exported_functions:
            func_copy = copy.copy(func)
            self_stub = self_func_stub(func_copy)
            setattr(self.compiled, func_name, self_stub)
            setattr(self.raw, func_name, self_stub)
            func_stub = error_stub(func_name)
            setattr(self.__class__, func_name, func_stub)
class TheanoModelLoss(AbstractModelLoss):
    import theano
    import theano.tensor as T
    import lasagne
    
    def __init__(self, **kwargs):        
        super().__init__(**kwargs)
        self.__build_theta_data_symbols()
        self._compile_and_disable_exported_functions()
        self._build()
    
    def fit(self, data, n_max_steps=100):        
        ## theano => no need to pass shared vars there, there are in the context
        self.__test_run(self.compiled.current_loss, self.compiled.grad, data)
        data_iterator = self.__warp_data_into_iterator(data)

        for i in range(n_max_steps+1): # to make it same as in autograd == n_max_steps
            data_batch = next(data_iterator)
            data_batch_record = to_record(data_batch)
            const_i, data_batch_record = self._test_data_shape(data_batch_record)            
            loss_val = self.compiled.perform_grad_step(**data_batch)
            theta_vals_record = record('WeightValues', 
                                       {name:shared_var.get_value(borrow=True)
                                        for name, shared_var in self.symbols['theta_shared_odict'].items()})
            info = {'n_iter': i}
            ret = self.step_callback(loss_val, theta_vals_record, data_batch_record, const_i, info)
            for behaviour in self.args.behaviours.values():
                behaviour.each_iteration(loss_val, theta_vals_record, data_batch_record, const_i, info)
            if ret == False:
                break
        
        self.behaviours = record('Behaviours', self.args.behaviours)
        return theta_vals_record
    
    def _build(self):
        if self.args.verbose:
            print(self.theano.config.mode)
            print('Building learning step function and gradient ..', end='')

            
        loss_expr = self.raw.loss(self.symbols['theta_nt'], self.symbols['data_nt'], 
                                  self.symbols['const_nt'])
        loss_grad_expr = self.T.grad(loss_expr, self.symbols['param_vars'])        
        updates = self.optimization_method(loss_grad_expr, self.symbols['param_vars'])
        self.compiled.perform_grad_step = self.theano.function(self.symbols['input_vars'], 
                                                               loss_expr, updates=updates)
        self.compiled.current_loss = self.theano.function(self.symbols['input_vars'], loss_expr)
        self.compiled.grad = self.theano.function(self.symbols['input_vars'], loss_grad_expr)
        
        if self.args.verbose:
            print('.. done')
            
        for func_name, func in self.raw.__dict__.items():
            if self.args.verbose:
                print('Building ', func_name, end='')

            # for export_data restrinctions
            local_input_vars = ([self.symbols['data_tensor_odict'][name] for name in func.data_list]
                                if func.data_list else self.symbols['input_vars'])
            
            local_full_input_vars = (list(self.symbols['theta_tensor_odict'].values()) 
                                     + local_input_vars)
            func_expr = func(self.symbols['theta_as_data_nt'], 
                             self.symbols['data_nt'], 
                             self.symbols['const_nt'])
            func_compiled = self.theano.function(local_full_input_vars, func_expr)
            func_compiled_warped = self.__warp_theano_function(func_compiled, data_list=func.data_list)
            setattr(self.compiled, func_name, func_compiled_warped)

            if self.args.verbose:
                print('.. done')

    def _optimization_method_builder(self, name, params):
        clip_val = params.pop('clip', None)
        clipped = ((lambda grads: [self.T.clip(grad, -clip_val, clip_val) for grad in grads])
                   if clip_val else (lambda grad: grad))
        # external API name: self.internal name
        kw_universal_mapping = {'learning_rate': 'learning_rate'}
        our_kwargs = {kw_universal_mapping[name]:value for name, value in params.items()}
        methods = {
            'adam': lambda grads, params: self.lasagne.updates.adam(clipped(grads), params, **our_kwargs)
        }
        return methods[name]
    
    def __build_theta_data_symbols(self):
        d = {}
        d['theta_nt'], d['theta_shared_odict'] = self.__build_initial_theta_shared_dict()
        d['data_nt'], d['data_tensor_odict'] = self.__build_data_tensor_dict()
        d['theta_as_data_nt'], d['theta_tensor_odict'] = self.__build_theta_tensor_dict()
        tensor_shapes = [x.shape for x in d['data_tensor_odict'].values()]
        
        ## actually just a view of data_nt, does not need to be passed to theano.function
        d['const_nt'] = extract_const_vars_from_tensors(dict(zip(tensor_shapes, self.d_shapes)))
        
        d['input_vars'] = list(d['data_tensor_odict'].values())
        d['param_vars'] = list(d['theta_shared_odict'].values())
        
        self.symbols = d
    
    def __build_data_tensor_dict(self):
        dim_type_mapping = {0: self.T.scalar, 1: self.T.vector, 2:self.T.matrix, 
                            3:self.T.tensor3, 4:self.T.tensor4}
        data_tensor_odict = collections.OrderedDict()
        for name, shape, dtype in zip(self.d_names, self.d_shapes, self.d_dtypes):
            dim_i = len(shape)
            if dim_i == 1 and shape[0] == 1: # scalar
                dim_i = 0
            try:
                tensor_type = dim_type_mapping[dim_i]
            except KeyError:
                raise ValueError("We do not support arrays of dimentions != [0..4], sorry :(")
            data_tensor_odict[name] = tensor_type(name, dtype=dtype)
        data_nt = record('DataTensors', data_tensor_odict)
        return data_nt, data_tensor_odict
    
    def __build_theta_tensor_dict(self):
        ## same but for THETA AS DATA input for self.compiled.func(theta, ...) calls
        dim_type_mapping = {0: self.T.scalar, 1: self.T.vector, 2:self.T.matrix, 
                            3:self.T.tensor3, 4:self.T.tensor4}
        theta_tensor_odict = collections.OrderedDict()
        for name, shape in zip(self.w_names, self.w_shapes):
            dim_i = len(shape)                
            try:
                tensor_type = dim_type_mapping[dim_i]
            except KeyError:
                raise ValueError("We do not support arrays of dimentions != [0..4], sorry :(")
            theta_tensor_odict[name] = tensor_type(name)
        theta_as_data_nt = record('ModelParametrTensors', theta_tensor_odict)
        return theta_as_data_nt, theta_tensor_odict

    def __build_initial_theta_shared_dict(self):
        theta_shared_odict = collections.OrderedDict()
        for name, shape, init_func in zip(self.w_names, self.w_shapes, self.w_inits):
            init_value = init_func(*shape)
            shared_var = self.theano.shared(init_value, name=name)
            theta_shared_odict[name] = shared_var
        
        theta_nt = record('ModelParameterSharedVars', theta_shared_odict)
        return theta_nt, theta_shared_odict
    
    def __warp_theano_function(self, th_func, data_list = None):
        def func(theta, data, const):
            input_dict = {}
            input_dict.update({w_name:getattr(theta, w_name).astype(self.theano.config.floatX) 
                               for w_name in self.w_names})
            input_dict.update({d_name:getattr(data, d_name) for d_name in (data_list or self.d_names)})
            ret_val = th_func(**input_dict)
            return ret_val
        return func
    
    def __warp_data_into_iterator(self, data):
        needs_warp = isinstance(data, dict) or not isinstance(data, collections.abc.Iterable)
        data_seq = [data] if needs_warp else data
        return itertools.cycle(data_seq)
    
    def _compile_and_disable_exported_functions(self):
        self.compiled = SimpleNamespace()
        self.raw = SimpleNamespace()
        
        def method_stub(for_name):
            def stub(*argv, **kwargs):
                raise RuntimeError("You shouldn't call model.{0} in user code"
                                   ", use model.compiled.{0}; original function "
                                   "is still avaliable at self.raw.{0}".format(for_name))
            return stub
        
        def new_func_stub(func):
            # funcs are just function (unbonded) from class
            # stub lives in namespace and does not get self
            # but the original method __class__.func needs self
            def stub(*argv, **kwargs):
                return func(self, *argv, **kwargs)
            stub.data_list = getattr(func, 'data_list', None)
            return stub
        
        ## in case of autograd we don't do anything actually
        ## we just forward same calls into self.compiled.func_name 
        for func_name, func in self._exported_functions:
            setattr(self.__class__, func_name, method_stub(func_name))
            setattr(self.raw, func_name, new_func_stub(func))
    
    def __test_run(self, loss_func, grad_func, data):
        data_pack = next(self.__warp_data_into_iterator(data))
        if self.args.test_forward_run:
            try:
                loss_func(**data_pack)
            except:
                print("Error while testing forward pass!")
                raise

            try:
                grad_func(**data_pack)
            except:
                print("Error while testing backward pass!")
                raise
class OpsBundle:
    def __init__(self, bk):
        self.bk = bk
        
    def softmax(self, A, axis=-1):
        # theano does not support elipses; above same as [..., newaxis]
        dim_new_axis =[slice(None, None), ]*(A.ndim)
        dim_new_axis[axis] = self.bk.newaxis
        expA = self.bk.exp(A - self.bk.max(A, axis=axis)[tuple(dim_new_axis)])
        s = self.bk.sum(expA, axis=axis)
        return expA / s[dim_new_axis]
    
class BaseBackend:
    def __init__(self, lookup_object):
        self.__lookup_object = lookup_object
        self.ops = OpsBundle(self)
        
    @property
    def bk(self):
        return self.__lookup_object
        
    # decorator
    @staticmethod
    def export(func):
        func.compilation_required = True
        return func

    @staticmethod
    def export_data(data_list):
        def tmp(func):
            func.compilation_required = True
            func.data_list = data_list
            return func
        return tmp
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.__lookup_object, name)

class AutoGradBackend(BaseBackend):
    def __init__(self):
        import autograd.numpy as __bk
        super().__init__(__bk)
        self.__dict__.update({
            'ModelLoss': AutogradModelLoss,
            'assert_arr_shape': assert_arr_shape
        })
        self.ModelLoss.bk = self # now in Loss we can access self.bk
        
    @staticmethod
    def eq(a, b):
        return a == b

class TheanoBackend(BaseBackend):
    import sys
    import theano
    import theano.tensor as T
    def __init__(self, mode='FAST_COMPILE'):
        self.sys.setrecursionlimit(100000)
        self.theano.config.mode = mode
        super().__init__(self.T)
        
        self.__dict__.update({
            'ModelLoss': TheanoModelLoss,
            'abs': abs,
            'newaxis': None,
            'assert_arr_shape': extract_const_vars_from_tensors,
        })
        
        self.ModelLoss.bk = self
        
    @classmethod
    def eq(cls, a, b):
        return cls.T.eq(a, b)
        
    @classmethod
    def sign(cls, a):
        return cls.T.sgn(a)
    
    @classmethod
    def multiply(cls, a, b):
        return a*b
    
    @classmethod
    def indices(cls, shape):
        mgrid_input = [slice(0, sh) for sh in shape]
        return cls.T.mgrid[mgrid_input]
class UnitBundle:
    def __init__(self, bk, name, weight_template, data_template):
        self.__name = name
        self.__prefix = name+'__'
        self.__template = weight_template
        self.global_template = {self.__prefix+key:val for key, val in weight_template.items()}
        self.data_template = data_template
        self.global_keys = set(self.global_template.keys())
        self.registered = False
        self.bk = bk
    
    def apply(self, global_theta, data_dict, **kwargs):
        if not self.registered:
            raise RuntimeError("The %s bundle was not registered via weight_bundles" % self.__name)
        local_theta = {
            key[len(self.__prefix):]:val # adapt - shortened key
            for key, val in global_theta._asdict().items()
            if key in self.global_keys
        }
        local_theta_nt = record('LocalBundleWeights', local_theta)
        
        d_names = self.data_template.keys()
        if set(d_names) != set(data_dict.keys()):
            raise ValueError("Exported %s data keys, got %s" 
                             % (set(d_names), set(data_dict.keys())))
        true_shapes = [data_dict[name].shape for name in d_names]
        required_shapes = [self.data_template[name] for name in d_names]
        local_const = self.bk.assert_arr_shape(dict(zip(true_shapes, required_shapes)))
        local_data_nt = record('LocalBundleData', data_dict)
        return self._func(local_theta_nt, local_data_nt, local_const, **kwargs)
    
    @property
    def weight_template(self):
        return self.global_template
    
    @abc.abstractmethod
    def _func(self): # may have whatever interface!
        pass