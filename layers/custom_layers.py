from lasagne.layers import Layer

class SelectOutputLayer(Layer):

    def __init__(self,incoming,index,**kwargs):

        # Initialize parent layer
        super(SelectOutputLayer, self).__init__(incoming, **kwargs)
        #self.incoming = incoming
        self.index = index

    def get_output_shape_for(self, input_shape):

        return input_shape[self.index]

    def get_output_for(self, input, **kwargs):
        return input[self.index]