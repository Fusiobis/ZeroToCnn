import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        #self.filters = np.random.randn(num_filters, 3, 3) / 9
        self.filters = np.random.randn(num_filters, 3, 3, 3) / 27
        self.biases = np.zeros(num_filters)

    def iterate_regions(self, image):
        #h, w = image.shape
        h, w, _ = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                #im_region = image[i:(i+3), j:(j+3)]
                im_region = image[i:(i+3), j:(j+3), :]
                yield im_region, i, j

    def backprop(self, d_L_d_out, learn_rate):
        #d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_filters = np.zeros_like(self.filters)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        d_L_d_bias = d_L_d_out.sum(axis=(0, 1))
        self.biases -= learn_rate * d_L_d_bias
        self.filters -= learn_rate * d_L_d_filters
        return None
    def forward(self, input):
        self.last_input = input
        #h,w = input.shape
        h,w,_ = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            #output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2, 3)) + self.biases
        
        return output