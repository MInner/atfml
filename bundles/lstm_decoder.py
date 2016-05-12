
# defined in atfml/examples/Char_LSTM.ipynb
from atfml.core import UnitBundle
from atfml.utils import inits

from atfml.bundles.lstm_encoder import LSTMEncoder

class LSTMDecoder(UnitBundle):
    def __init__(self, bk, name, n_hidden_dim, n_input_dim, n_steps, output_to_input_connection=True):
        self.n_steps = n_steps
        self.n_hidden_dim = n_hidden_dim
        self.o_to_i = output_to_input_connection
        
        w_width = n_input_dim + n_hidden_dim
        arg_dict = {
            'name': name,
            'weight_template': {
                'x_0': {'shape': (n_input_dim, ) },
                'c_0': {'shape': (n_hidden_dim, ) },
                'W_i': {'shape': (w_width, n_hidden_dim), 'init_method': inits.identity_repeat_init },
                'b_i': {'shape': (n_hidden_dim, ) },
                'W_c': {'shape': (w_width, n_hidden_dim), 'init_method': inits.identity_repeat_init },
                'b_c': {'shape': (n_hidden_dim, ) },
                'W_f': {'shape': (w_width, n_hidden_dim), 'init_method': inits.identity_repeat_init },
                'b_f': {'shape': (n_hidden_dim, ) },
                'W_o': {'shape': (w_width, n_hidden_dim), 'init_method': inits.identity_repeat_init },
                'b_o': {'shape': (n_hidden_dim, ) },
                'V_c': {'shape': (n_hidden_dim, n_hidden_dim), 
                        'init_method': inits.identity_repeat_init },
            },
            'data_template': {
                'X_expected': ('batch_size', self.n_steps, n_input_dim),
                'H_0': ('batch_size', n_hidden_dim),
            }
        }
        super().__init__(bk, **arg_dict)
        
    def _func(self, theta, data, const):
        """ same as LSTM as in LSTMBundle """
        np = None
        bk = self.bk
        h_prev = data.H_0
        c_prev = bk.repeat(theta.c_0[bk.newaxis, :], const.batch_size, axis=0)
        X_0 = bk.repeat(theta.x_0[bk.newaxis, :], const.batch_size, axis=0)
        if self.o_to_i:
            X_effective = bk.concatenate([X_0[:, bk.newaxis, :], data.X_expected[:, :-1, :]], axis=1)
        else:
            X_effective = bk.repeat(X_0[:, bk.newaxis, :], self.n_steps, axis=1)
      
        decoded_list = []
        sigmoid = lambda x: x/(1+bk.abs(x+1e-20))
        for t in range(self.n_steps):
            x_t = X_effective[:, t, :]
            concat_h_x = bk.concatenate([h_prev, x_t], axis=1)
            i_t = sigmoid(bk.dot(concat_h_x, theta.W_i) + theta.b_i[bk.newaxis, :])
            c_tld_t = sigmoid(bk.dot(concat_h_x, theta.W_c) + theta.b_c[bk.newaxis, :])
            f_t = sigmoid(bk.dot(concat_h_x, theta.W_f) + theta.b_f[bk.newaxis, :])
            c_t = bk.multiply(i_t, c_tld_t) + bk.multiply(f_t, c_prev)
            o_t = sigmoid(bk.dot(concat_h_x, theta.W_o) + bk.dot(c_t, theta.V_c) 
                          + theta.b_o[bk.newaxis, :])
            h_t = bk.multiply(o_t, bk.tanh(c_t))
            h_prev = h_t
            c_prev = c_t
            
            decoded_list.append(h_prev)
            
        decoded = bk.concatenate([x[:, bk.newaxis, :] for x in decoded_list], axis=1)
        bk.assert_arr_shape({decoded.shape: (const.batch_size, self.n_steps, self.n_hidden_dim)})
        
        return decoded