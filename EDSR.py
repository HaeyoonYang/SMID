import torch.nn as nn


def basic_conv(in_channels, out_channels, ker, bias=True):
    return nn.Conv2d(in_channels, out_channels, ker, padding=((ker//2), (ker//2)), bias=bias)


class EDSR(nn.Module):
    def __init__(self, depth=16, out_channels=96, conv=basic_conv):
        super(EDSR, self).__init__()
        self.out_ch = out_channels
        self.pad = (1, 1)
        self.depth = depth

        ker = 3
        act = nn.ReLU(True)

        m_head = [conv(12, out_channels, ker)]
        m_body = [ResBlock(conv, out_channels, ker) for _ in range(depth-4)]
        m_body.append(conv(out_channels, out_channels, ker))
        m_tail = [conv(out_channels, out_channels, ker),
                  conv(out_channels, 3, ker)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, noisy_imgs):
        x = self.head(noisy_imgs)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


class ResBlock(nn.Module):
    def __init__(self, conv, out_ch, ker, bias=True, act=nn.ReLU(True)):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(out_ch, out_ch, ker, bias=bias))
        m.append(act)
        m.append(conv(out_ch, out_ch, ker, bias=bias))

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

