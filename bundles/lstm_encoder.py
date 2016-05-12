
# defined in atfml/examples/Char_LSTM.ipynb
from atfml.core import UnitBundle
from atfml.utils import inits

class LSTMEncoder(UnitBundle):
    def __init__(self, bk, name, n_hidden_dim, n_input_dim, n_steps):
        self.n_steps = n_steps
        w_width = n_input_dim + n_hidden_dim
        
        arg_dict = {
            'name': name,
            'weight_template': {
                'h_0': {'shape': (n_hidden_dim, ) },
                'c_0': {'shape': (n_hidden_dim, ) },
                'W_i': {'shape': (w_width, n_hidden_dim), 'init_method': inits.identity_repeat_init },
                'b_i': {'shape': (n_hidden_dim, ) },
                'W_c': {'shape': (w_width, n_hidden_dim), 'init_method': inits.identity_repeat_init },
                'b_c': {'shape': (n_hidden_dim, ) },
                'W_f': {'shape': (w_width, n_hidden_dim), 'init_method': inits.identity_repeat_init },
                'b_f': {'shape': (n_hidden_dim, ) },
                'W_o': {'shape': (w_width, n_hidden_dim), 'init_method': inits.identity_repeat_init },
                'b_o': {'shape': (n_hidden_dim, ) },
                'V_c': {'shape': (n_hidden_dim, n_hidden_dim), 'init_method': inits.identity_repeat_init },
            },
            'data_template': {
                'X': ('batch_size', 'seq_len', n_input_dim),
            }
        }
        
        super().__init__(bk, **arg_dict)
        
    def _func(self, theta, data, const):
        """
            i_t:[batch_size, hidden_dim]
                = sigmoid( <h_{t-1}, x_t>:[batch_size, hid_dim + input_dim ] @ W_i:[hid+input, hid] + b_i[1, hid])
            c_tilde_t:[batch_size, hidden_dim]
                = tanh( <h_{t-1}, x_t>:[batch_size, hid_dim + input_dim ] @ W_c:[hid+input, hid] + b_c[1, hid])
            f_t:[batch_size, hidden_dim]
                = sigmoid( <h_{t-1}, x_t>:[batch_size, hid_dim + input_dim ] @ W_f:[hid+input, hid] + b_f[1, hid])
            c_t:[batch_size, hidden_dim]
                = i_t*c_tilde_t + f_t*c_{t-1}
            o_t:[batch_size, hidden_dim]
                = sigmoid( <h_{t-1}, x_t>:[batch_size, hid_dim + input_dim ] @ W_o:[hid+input, hid] 
                            + c_t @ V_c:[hid, hid]
                            + b_o[1, hid])
            h_t:[batch_size, hidden_dim]
                = o_t*tanh(c_t)
        """
        bk = self.bk
        h_prev = self.bk.repeat(theta.h_0[bk.newaxis, :], const.batch_size, axis=0)
        c_prev = self.bk.repeat(theta.c_0[bk.newaxis, :], const.batch_size, axis=0)
        
        sigmoid = lambda x: x/(1+bk.abs(x+1e-20))
        for t in range(self.n_steps):
            x_t = data.X[:, t]
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
            
        return h_prev