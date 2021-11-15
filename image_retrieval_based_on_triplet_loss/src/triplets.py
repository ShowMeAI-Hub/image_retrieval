import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class TripletStream:
    
    def __init__(self, streams, batch_size):
        self.classes = [c for c in streams]
        self.streams = streams
        self.batch_size = batch_size
        self.buffers = {c: [] for c in streams}

        self.random_deltas = np.arange(1, len(streams))

    def pop(self, c):
        if not self.buffers[c]:
            batch_data, batch_labels = next(self.streams[c])
            self.buffers[c] = [x for x in batch_data]

        return self.buffers[c].pop()

    def __iter__(self):
        return self

    def __next__(self):
        # 图像类别 与 其他随机类别
        orig_classes = np.random.choice(self.classes, (self.batch_size,))
        comp_classes = np.random.choice(self.classes, (self.batch_size,))

        unintended_matches = orig_classes == comp_classes
        n_matches = sum(unintended_matches)
        if n_matches:
            comp_classes[unintended_matches] = (
                    comp_classes[unintended_matches] +
                    np.random.choice(self.random_deltas, n_matches)
                    ) % len(self.classes)

        # x1为同类别，x2位不同类别
        x  = [self.pop(c) for c in orig_classes]
        x1 = [self.pop(c) for c in orig_classes]
        x2 = [self.pop(c) for c in comp_classes]

        samples = {
                'x': np.array(x),
                'x1': np.array(x1),
                'x2': np.array(x2),
        }

        labels = np.ones(self.batch_size)
        return samples, labels


class TripletGenerator:
'''
三元组生成器
'''
    def __init__(self):
        self.gen = ImageDataGenerator(
                #preprocessing_function=lambda x:
                    #x.astype('float32') / 255.
                featurewise_center=True,
                featurewise_std_normalization=True,
                horizontal_flip=True,
        )
            
    def flow(self, x, y, indices=None, batch_size=64):
        if not indices:
            classes = np.unique(y)
            indices = {c: np.where(y == c)[0] for c in classes}

        streams = {
            c: self.gen.flow(
                x[matching_indices, :],
                y[matching_indices, :],
                batch_size=batch_size
            ) for c, matching_indices in indices.items()
        }

        return TripletStream(streams, batch_size)
