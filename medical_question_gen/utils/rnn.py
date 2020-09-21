import torch

class StackedBiLSTM(torch.nn.Module):
    def __init__(self, *,
                 lstm_layer_num: int = 3,
                 input_feature_len: int,
                 output_feature_len: int,
                 hidden_size: int,
                 bidirectional: bool = True,
                 dropout: float = 0.1,
                 sequence_len: int = None):
        super().__init__()
        self.direction = 2 if bidirectional else 1
        self.sequence_len = sequence_len

        self.lstm_layer = torch.nn.LSTM(input_feature_len,
                                        hidden_size,
                                        num_layers=lstm_layer_num,
                                        batch_first=True,
                                        bidirectional=bidirectional,
                                        dropout=dropout)
        self.linear_layer = torch.nn.Linear(hidden_size * self.direction, output_feature_len)
        self.cuda()

    def forward(self, x):
        # 这里r_out shape永远是(seq, batch, output_size)，与网络的batch_first参数无关
        lstm_layer_out, _ = self.lstm_layer(x, None)
        out = self.linear_layer(lstm_layer_out)
        return out

    def set_train(self):
        self.train()

    def set_eval(self):
        self.eval()


class SimpleLSTM(StackedBiLSTM):
    def __init__(self, input_feature_len: int, hidden_size: int, output_feature_len: int, sequence_len: int = None):
        super().__init__(lstm_layer_num=1,
                         input_feature_len=input_feature_len,
                         output_feature_len=output_feature_len,
                         hidden_size=hidden_size,
                         bidirectional=False,
                         dropout=0,
                         sequence_len=sequence_len)


class GRUEncoder(torch.nn.Module):
    def __init__(self, *,
                 gru_layer_num: int = 1,
                 input_feature_len: int,
                 sequence_len: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 device='cuda'):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.lstm_layer_num = gru_layer_num
        self.direction = 2 if bidirectional else 1

        self.gru_layer = torch.nn.GRU(input_feature_len,
                                      hidden_size,
                                      num_layers=gru_layer_num,
                                      batch_first=True,
                                      bidirectional=bidirectional,
                                      dropout=dropout)

        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, x):
        gru_output, h_n = self.gru_layer(x)

        # lstm_output sum-reduced by direction
        gru_output = gru_output.view(x.size(0), self.sequence_len, self.direction, self.hidden_size)
        gru_output = gru_output.sum(2)

        # lstm_states sum-reduced by direction
        h_n = h_n.view(self.lstm_layer_num, self.direction, x.size(0), self.hidden_size)
        h_n = h_n.sum(1)

        return gru_output, h_n

    def init_hidden(self, batch_size: int):
        h_0 = torch.zeros(self.lstm_layer_num * self.direction, batch_size, self.hidden_size, device=self.device)
        return h_0


class BahdanauAttention(torch.nn.Module):
    def __init__(self, *,
                 hidden_size: int,
                 units: int,
                 device="cuda",
                 mode_source: str = 'custom'):
        super().__init__()
        if mode_source not in ('custom', 'nlp'):
            raise NotImplementedError
        self.mode_source = mode_source
        self.W1 = torch.nn.Linear(hidden_size, units)
        self.W2 = torch.nn.Linear(hidden_size, units)
        self.V = torch.nn.Linear(units, 1)

        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, encoder_output, last_layer_h_n):
        score = self.V(torch.tanh(self.W1(encoder_output) + self.W2(last_layer_h_n.unsqueeze(1))))
        attention_weights = torch.nn.functional.softmax(score, 1)
        if self.mode_source == 'mode_source':
            context_vector = attention_weights * encoder_output
            context_vector = context_vector.sum(1)
        else:
            context_vector = None
        """
        ax = series(context_vector.squeeze().detach().cpu().numpy(), label='context_vector')
        series(last_layer_h_n.squeeze().detach().cpu().numpy(), ax=ax, label='last_layer_h_n')
        series(attention_weights.squeeze().detach().cpu().numpy(), title='attention_weights')
        """
        tt = 1
        return context_vector, attention_weights


class GRUDecoder(torch.nn.Module):
    def __init__(self, *,
                 gru_layer_num: int = 1,
                 decoder_input_feature_len: int,
                 output_feature_len: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 dropout: float = 0.1,
                 attention_units: int = 128,
                 mode_source: str = 'custom',
                 device="cuda"):
        super().__init__()
        if mode_source not in ('custom', 'nlp'):
            raise NotImplementedError
        else:
            self.mode_source = mode_source
        self.lstm_layer_num = gru_layer_num
        self.direction = 2 if bidirectional else 1
        self.hidden_size = hidden_size

        self.gru_layer = torch.nn.GRU(decoder_input_feature_len,
                                      hidden_size,
                                      num_layers=gru_layer_num,
                                      batch_first=True,
                                      bidirectional=bidirectional,
                                      dropout=dropout)

        self.attention = BahdanauAttention(
            hidden_size=hidden_size,
            units=attention_units
        )  # type: BahdanauAttention

        self.out = torch.nn.Linear(hidden_size, output_feature_len)
        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, *, y, encoder_output, h_n, this_time_step: int = None):
        # Only use the hidden information from the last layer for attention
        context_vector, attention_weights = self.attention(encoder_output, h_n[-1])
        if self.mode_source == 'nlp':
            y = torch.cat((context_vector.unsqueeze(1), y.unsqueeze(1)), -1)
        ###############################################################################################################
        else:
            attention_weights = (attention_weights - attention_weights.min(1, keepdim=True).values) / (
                    attention_weights.max(1, keepdim=True).values - attention_weights.min(1, keepdim=True).values)
            y = encoder_output * attention_weights
        gru_output, h_n = self.gru_layer(y[:, this_time_step, :].unsqueeze(1), h_n)
        ###############################################################################################################
        # else:
        #     where_max = attention_weights.argmax(1)
        #     y = torch.zeros((encoder_output.size(0), 1, encoder_output.size(2)), device=self.device)
        #     for this_batch_index in range(where_max.size(0)):
        #         y[this_batch_index] = encoder_output[this_batch_index, where_max[this_batch_index], :]
        # gru_output, h_n = self.gru_layer(y, h_n)
        ###############################################################################################################

        output = self.out(gru_output.squeeze(1))
        return output, h_n, attention_weights

    def init_hidden(self, batch_size: int):
        h_0 = torch.zeros(self.lstm_layer_num * self.direction, batch_size, self.hidden_size, device=self.device)
        return h_0


class GRUEncoderDecoderWrapper(torch.nn.Module):
    def __init__(self, *, gru_encoder: GRUEncoder,
                 gru_decoder: GRUDecoder,
                 output_sequence_len: int,
                 output_feature_len: int,
                 teacher_forcing: float = 0.001,
                 device="cuda",
                 mode_source: str = 'custom'):
        super().__init__()
        if mode_source not in ('custom', 'nlp'):
            raise NotImplementedError
        else:
            self.mode_source = mode_source
        self.gru_encoder = gru_encoder  # type: GRUEncoder
        self.gru_decoder = gru_decoder  # type: GRUDecoder
        self.output_sequence_len = output_sequence_len
        self.output_feature_len = output_feature_len
        self.teacher_forcing = teacher_forcing
        self.device = device
        if "cuda" in device:
            self.cuda()

    def forward(self, x, y=None):
        encoder_output, encoder_h_n = self.gru_encoder(x)
        # if self.mode_source == 'custom':
        #     decoder_h_n = self.gru_decoder.init_hidden(x.size(0))
        # else:
        decoder_h_n = encoder_h_n
        outputs = torch.zeros((x.size(0), self.output_sequence_len, self.output_feature_len),
                              device=self.device)
        this_time_step_decoder_output = None
        for i in range(self.output_sequence_len):
            if i == 0:
                this_time_step_decoder_input = torch.zeros((x.size(0), self.output_feature_len),
                                                           device=self.device)
            else:
                if (y is not None) and (torch.rand(1) < self.teacher_forcing):
                    this_time_step_decoder_input = y[:, i - 1, :]
                else:
                    this_time_step_decoder_input = this_time_step_decoder_output

            this_time_step_decoder_output, decoder_h_n, _ = self.gru_decoder(
                y=this_time_step_decoder_input,
                encoder_output=encoder_output,
                h_n=decoder_h_n,
                this_time_step=i
            )

            outputs[:, i, :] = this_time_step_decoder_output

        return outputs

    def set_train(self):
        self.gru_encoder.train()
        self.gru_decoder.train()
        self.train()

    def set_eval(self):
        self.gru_encoder.eval()
        self.gru_decoder.eval()
        self.eval()
