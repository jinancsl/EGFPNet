import torch
import torch.nn as nn

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class up_conv_skip(nn.Module):
    def __init__(self, in_ch, out_ch, skip=2):
        super(up_conv_skip, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=skip),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv_EGFP(nn.Module):
    def __init__(self, in_ch, out_ch, skip):
        super(down_conv_EGFP, self).__init__()
        self.up = nn.Sequential(
            nn.MaxPool2d(skip, skip, ceil_mode=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class EGFPNet(nn.Module):
    """
    EGFPNet - Basic Implementation
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(EGFPNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.skip_e5_to_edge = up_conv_skip(filters[4], filters[1], skip=8)

        self.conv2_edge = conv_block(filters[1] * 2, filters[1])
        self.Up_edge = up_conv(filters[1], filters[0])
        self.Conv_edge = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # edge to e5 d5 d4 d3
        # e5
        self.edge_to_e5 = down_conv_EGFP(filters[1], filters[4], skip=8)
        self.e5_and_edge_feature = conv_block(filters[4] * 2, filters[4])
        self.e5_and_edge_to_d5_feature = up_conv_skip(filters[4], filters[3], skip=2)
        self.e5_and_edge_to_out_feature = up_conv_skip(filters[4], filters[0], skip=16)
        self.e5_and_edge_to_out = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # d5
        self.edge_to_d5 = down_conv_EGFP(filters[1], filters[3], skip=4)
        self.d5_and_edge_feature = conv_block(filters[3] * 3, filters[3])
        self.d5_and_edge_to_d4_feature = up_conv_skip(filters[3], filters[2], skip=2)
        self.d5_and_edge_to_out_feature = up_conv_skip(filters[3], filters[0], skip=8)
        self.d5_and_edge_to_out = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # d4
        self.edge_to_d4 = down_conv_EGFP(filters[1], filters[2], skip=2)
        self.d4_and_edge_feature = conv_block(filters[2] * 3, filters[2])
        self.d4_and_edge_to_d3_feature = up_conv_skip(filters[2], filters[1], skip=2)
        self.d4_and_edge_to_out_feature = up_conv_skip(filters[2], filters[0], skip=4)
        self.d4_and_edge_to_out = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # d3
        self.d3_and_edge_feature = conv_block(filters[1] * 3, filters[1])
        self.d3_and_edge_to_out_feature = up_conv_skip(filters[1], filters[0], skip=2)
        self.d3_and_edge_to_out = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        #final out
        self.final = conv_block(filters[0] * 4, filters[0])
        self.Conv_final = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        e5_to_e2 = self.skip_e5_to_edge(e5)
        e2_edge = self.conv2_edge(torch.cat((e2, e5_to_e2), dim=1))
        e2_edge_ = self.Up_edge(e2_edge)
        out_edge = self.Conv_edge(e2_edge_)

        #edge to e5 d5 d4 d3
        # e5
        edge_to_e5 = self.edge_to_e5(e2_edge)
        e5_and_edge_feature = self.e5_and_edge_feature(torch.cat((e5, edge_to_e5), dim=1))
        e5_and_edge_to_d5_feature = self.e5_and_edge_to_d5_feature(e5_and_edge_feature)
        e5_and_edge_to_out_feature = self.e5_and_edge_to_out_feature(e5_and_edge_feature)
        e5_and_edge_to_out = self.e5_and_edge_to_out(e5_and_edge_to_out_feature)

        # d5
        edge_to_d5 = self.edge_to_d5(e2_edge)
        d5_and_edge_feature = self.d5_and_edge_feature(torch.cat((d5, e5_and_edge_to_d5_feature, edge_to_d5), dim=1))
        d5_and_edge_to_d4_feature = self.d5_and_edge_to_d4_feature(d5_and_edge_feature)
        d5_and_edge_to_out_feature = self.d5_and_edge_to_out_feature(d5_and_edge_feature)
        d5_and_edge_to_out = self.d5_and_edge_to_out(d5_and_edge_to_out_feature)

        # d4
        edge_to_d4 = self.edge_to_d4(e2_edge)
        d4_and_edge_feature = self.d4_and_edge_feature(torch.cat((d4, d5_and_edge_to_d4_feature, edge_to_d4), dim=1))
        d4_and_edge_to_d3_feature = self.d4_and_edge_to_d3_feature(d4_and_edge_feature)
        d4_and_edge_to_out_feature = self.d4_and_edge_to_out_feature(d4_and_edge_feature)
        d4_and_edge_to_out = self.d4_and_edge_to_out(d4_and_edge_to_out_feature)

        # d3
        d3_and_edge_feature = self.d3_and_edge_feature(torch.cat((d3, d4_and_edge_to_d3_feature, e2_edge), dim=1))
        d3_and_edge_to_out_feature = self.d3_and_edge_to_out_feature(d3_and_edge_feature)
        d3_and_edge_to_out = self.d3_and_edge_to_out(d3_and_edge_to_out_feature)

        # final out
        out_final = self.final(torch.cat((e5_and_edge_to_out_feature, d5_and_edge_to_out_feature, d4_and_edge_to_out_feature, d3_and_edge_to_out_feature), dim=1))
        out_final = self.Conv_final(out_final)
        return out_final, out_edge, e5_and_edge_to_out, d5_and_edge_to_out, d4_and_edge_to_out, d3_and_edge_to_out





