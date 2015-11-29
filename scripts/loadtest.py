
import lasagne
import voxnet

dims = (16,16,16)
n_channels = 1
n_classes = 10

model = voxnet.models.build_single(dims, n_channels, n_classes)

layers = lasagne.layers.get_all_layers(model['l_out'])
for layer in layers:
    print layer
