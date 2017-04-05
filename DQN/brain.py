from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self._make_network()

    def _make_network(self):
        model = Sequential()
        model.add(Dense(input_dim=self.num_states, activation='relu', output_dim=64))
        model.add(Dense(activation='linear', output_dim=self.num_actions))
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        self.model = model

    def predict(self, s):
        return self.model.predict(s)

    def predict_one(self, s):
        return self.model.predict(s.reshape(1, self.num_states)).flatten()

    def train(self, x, y, epoch=0, verbose=False):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

