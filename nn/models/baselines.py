import clrs

_Features = clrs.Features


class UniformRandom:
    def predict(self, features: _Features):
        from numpy.random import rand

        for inp in features.inputs:
            if inp.location in [clrs.Location.NODE, clrs.Location.EDGE]:
                batch_size, num_nodes = inp.data.shape[0], inp.data.shape[1]
                break

        return rand(batch_size, num_nodes)


class NormalRandom:
    def predict(self, features: _Features):
        from numpy.random import randn

        for inp in features.inputs:
            if inp.location in [clrs.Location.NODE, clrs.Location.EDGE]:
                batch_size, num_nodes = inp.data.shape[0], inp.data.shape[1]
                break

        return randn(batch_size, num_nodes)
