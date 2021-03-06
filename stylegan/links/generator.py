from math import sqrt as root
from chainer import Parameter, Link, Chain
from chainer.functions import sqrt, sum, scale, convolution_2d, resize_images, broadcast_to, pad
from chainer.initializers import Zero, One, Normal
from stylegan.links.common import GaussianDistribution, EqualizedLinear, LeakyRelu

class StyleAffineTransform(Chain):

	def __init__(self, size, channels):
		super().__init__()
		with self.init_scope():
			self.s = EqualizedLinear(size, channels, initial_bias=One(), gain=1)

	def __call__(self, w):
		return self.s(w)

class WeightDemodulatedConvolution2D(Link):

	def __init__(self, in_channels, out_channels, pointwise=False, demod=True, gain=root(2)):
		super().__init__()
		self.demod = demod
		self.ksize = 1 if pointwise else 3
		self.pad = 0 if pointwise else 1
		self.c = gain * root(1 / (in_channels * self.ksize ** 2))
		with self.init_scope():
			self.w = Parameter(shape=(out_channels, in_channels, self.ksize, self.ksize), initializer=Normal(1.0))
			self.b = Parameter(shape=out_channels, initializer=Zero())

	def __call__(self, x, y):
		out_channels = self.b.shape[0]
		batch, in_channels, height, width = x.shape
		mod_w = self.w * y.reshape(batch, 1, in_channels, 1, 1)
		w = mod_w / sqrt(sum(mod_w ** 2, axis=(2, 3, 4), keepdims=True) + 1e-8) if self.demod else mod_w
		group_w = w.reshape(batch * out_channels, in_channels, self.ksize, self.ksize)
		group_x = x.reshape(1, batch * in_channels, height, width)
		pad_group_x = pad(group_x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="edge")
		h = convolution_2d(pad_group_x, group_w, stride=1, pad=0, groups=batch)
		return h.reshape(batch, out_channels, height, width) + self.b.reshape(1, out_channels, 1, 1)

class NoiseAdder(Chain):

	def __init__(self, channels):
		super().__init__()
		with self.init_scope():
			self.g = GaussianDistribution()
			self.s = Parameter(shape=channels, initializer=Zero())

	def __call__(self, x):
		b, _, h, w = x.shape
		n = broadcast_to(self.g(b, 1, h, w), x.shape)
		return x + scale(n, self.s)

class Upsampler(Link):

	def __call__(self, x):
		height, width = x.shape[2:]
		return resize_images(x, (height * 2, width * 2), align_corners=False)

class ToRGB(Chain):

	def __init__(self, in_channels):
		super().__init__()
		with self.init_scope():
			self.w = WeightDemodulatedConvolution2D(in_channels, 3, pointwise=True, demod=False, gain=1)

	def __call__(self, x, y):
		return self.w(x, y)

class InitialSkipArchitecture(Chain):

	def __init__(self, size, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.c1 = Parameter(shape=(in_channels, 4, 4), initializer=Normal(1.0))
			self.s1 = StyleAffineTransform(size, in_channels)
			self.w1 = WeightDemodulatedConvolution2D(in_channels, out_channels)
			self.n1 = NoiseAdder(out_channels)
			self.a1 = LeakyRelu()
			self.s2 = StyleAffineTransform(size, out_channels)
			self.trgb = ToRGB(out_channels)

	def __call__(self, w):
		batch = w.shape[0]
		h1 = self.c1.reshape(1, *self.c1.shape)
		h2 = broadcast_to(h1, (batch, *self.c1.shape))
		h3 = self.w1(h2, self.s1(w))
		h4 = self.n1(h3)
		h5 = self.a1(h4)
		return h5, self.trgb(h5, self.s2(w))

class SkipArchitecture(Chain):

	def __init__(self, size, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.up = Upsampler()
			self.s1 = StyleAffineTransform(size, in_channels)
			self.w1 = WeightDemodulatedConvolution2D(in_channels, out_channels)
			self.n1 = NoiseAdder(out_channels)
			self.a1 = LeakyRelu()
			self.s2 = StyleAffineTransform(size, out_channels)
			self.w2 = WeightDemodulatedConvolution2D(out_channels, out_channels)
			self.n2 = NoiseAdder(out_channels)
			self.a2 = LeakyRelu()
			self.s3 = StyleAffineTransform(size, out_channels)
			self.trgb = ToRGB(out_channels)
			self.skip = Upsampler()

	def __call__(self, x, y, w):
		h1 = self.up(x)
		h2 = self.w1(h1, self.s1(w))
		h3 = self.n1(h2)
		h4 = self.a1(h3)
		h5 = self.w2(h4, self.s2(w))
		h6 = self.n2(h5)
		h7 = self.a2(h6)
		return h7, self.skip(y) + self.trgb(h7, self.s3(w))
