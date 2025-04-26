class StandardScaler:

    def __init__(self, mean=None, std=None):
        
        self.mean = mean
        self.std = std


    def fit_transform(self, data):
        
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std


    def transform(self, data):
        
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        
        return (data * self.std) + self.mean