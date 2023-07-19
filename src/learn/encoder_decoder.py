"""
编码器和解码器
"""
from torch import nn


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口，将固定形状的编码状态映射到长度可变的序列"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, src, *args):
        # 指定长度可变的序列作为编码器的输入`src`
        raise NotImplementedError


class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, encode_outputs, *args):
        # 用于将编码器的输出（`encode_outputs`）转换为编码后的状态
        raise NotImplementedError

    def forward(self, src, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encode_input, decode_input, *args):
        enc_outputs = self.encoder(encode_input, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(decode_input, dec_state)

