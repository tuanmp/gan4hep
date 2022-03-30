"""
This is a simple MLP-base conditional GAN.
Note that the conditional input is not
given to the discriminator.
"""
from tensorflow import keras
from tensorflow.keras import layers

class GAN():
    def __init__(self,
        noise_dim: int = 4, gen_output_dim: int = 2,
        cond_dim: int = 0, **kwargs):
        """
        noise_dim: dimension of the noises
        gen_output_dim: output dimension
        cond_dim: in case of conditional GAN, 
                  it is the dimension of the condition
        """
        self.noise_dim = noise_dim
        self.gen_output_dim = gen_output_dim
        self.cond_dim = cond_dim

        self.gen_input_dim = self.noise_dim + self.cond_dim

        # Build the critic
        self.discriminator = self.build_critic(
            dense_layers=kwargs.get("disc_dense_layers", [256,256]),
            dropout_rate=kwargs.get('disc_dropout'))
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()


    def build_generator(self):
        gen_input_dim = self.gen_input_dim
        model = keras.Sequential([
            keras.Input(shape=(gen_input_dim,)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(self.gen_output_dim),
            layers.Activation("tanh"),
        ], name='Generator')
        return model

    def build_critic(self, dropout_rate=0., dense_layers=[256, 256]):
        gen_output_dim = self.gen_output_dim
        cond_dim = self.cond_dim

        model = keras.Sequential(
            name='Discriminator'
        )
        model.add(keras.Input(shape=(gen_output_dim + cond_dim,)))
        for dense_layer in dense_layers:
            model.add(layers.Dense(dense_layer))
            model.add(layers.Dropout(dropout_rate, seed=0))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("filename", help='input filename', default=None, nargs='+')
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--num-test-evts", help='number of testing events', default=10000, type=int)
    add_arg("--inference", help='perform inference only', action='store_true')
    add_arg("-v", '--verbose', help='tf logging verbosity', default='INFO',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    add_arg("--lr", help='learning rate', type=float, default=0.0001)
    args = parser.parse_args()

    logging.set_verbosity(args.verbose)

    gan = GAN()