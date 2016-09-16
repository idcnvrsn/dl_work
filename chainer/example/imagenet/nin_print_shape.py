import math

import chainer
import chainer.functions as F
import chainer.links as L


class NIN(chainer.Chain):

    """Network-in-Network example model."""

    insize = 227

    def __init__(self):
        w = math.sqrt(2)  # MSRA scaling
        super(NIN, self).__init__(
            mlpconv1=L.MLPConvolution2D(
                3, (96, 96, 96), 11, stride=4, wscale=w),
            mlpconv2=L.MLPConvolution2D(
                96, (256, 256, 256), 5, pad=2, wscale=w),
            mlpconv3=L.MLPConvolution2D(
                256, (384, 384, 384), 3, pad=1, wscale=w),
            mlpconv4=L.MLPConvolution2D(
                384, (1024, 1024, 10), 3, pad=1, wscale=w),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        print("input:",x.data.shape)
        self.clear()
        mc1 = self.mlpconv1(x)
        print("mlpconv1 result",mc1.data.shape)
        relu1 = F.relu(mc1)
        print("relu1 result",relu1.data.shape)
        h = F.max_pooling_2d(relu1, 3, stride=2)
        print("mpool1 result",h.data.shape)
        print("")
        
        mc2 = self.mlpconv2(h)
        print("mlpconv2 result",mc2.data.shape)
        relu2 = F.relu(mc2)
        print("relu2 result",relu2.data.shape)
        h = F.max_pooling_2d(relu2, 3, stride=2)
        print("mpool2 result",h.data.shape)
        print("")

#        h = F.max_pooling_2d(F.relu(self.mlpconv3(h)), 3, stride=2)
        mc3 = self.mlpconv3(h)
        print("mlpconv3 result",mc3.data.shape)
        relu3 = F.relu(mc3)
        print("relu3 result",relu3.data.shape)
        h = F.max_pooling_2d(relu3, 3, stride=2)
        print("mpool3 result",h.data.shape)
        print("")
        
#        h = self.mlpconv4(F.dropout(h, train=self.train))
        dr = F.dropout(h, train=self.train)
        print("dr result",dr.data.shape)
        h = self.mlpconv4(dr)
        print("mlpconv4 result",h.data.shape)
        print("")

#        h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))
        ap = F.average_pooling_2d(h, 6)
        print("ap result",ap.data.shape)
        print("")
        
        h = F.reshape(ap, (x.data.shape[0], 10))

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss
