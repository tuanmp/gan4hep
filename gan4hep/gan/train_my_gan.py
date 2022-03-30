#!/user/bin/env python
from datetime import datetime
import os
import json
import tqdm
import numpy as np

from scipy import stats
import tensorflow as tf
# tf.random.set_seed(
#     0
# )
# np.random.seed(0)


from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices("GPU")
logging.info("found {} GPUs".format(len(gpus)))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
from gan import GAN
# from aae import AAE
# from cgan import CGAN
# from wgan import WGAN

all_gans = ['GAN', "AAE", 'CGAN', 'WGAN']

from gan4hep.preprocess import herwig_angles
from gan4hep.preprocess import herwig_angles2
from gan4hep.preprocess import dimuon_inclusive
from gan4hep.preprocess import geant4_leading_products
from gan4hep.preprocess import geant4_momentum_transfer
from gan4hep.preprocess import geant4_COM_frame
from gan4hep.preprocess import geant4_momentum_transfer_delta_phi, geant4_momentum_transfer_delta_phi_eta, geant4_COM_delta_phi_eta_E_pT, geant4_COM_eta_E_pT

from utils import evaluate, log_metrics
from gan4hep.utils_plot import compare

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return tf.reduce_mean(total_loss)

def generator_loss(fake_output):
    return tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))

def train(train_truth, test_truth, model, gen_lr, disc_lr, batch_size,
    max_epochs, log_dir, xlabels, config, disable_tqdm=False, train_in=None, test_in=None, train_from_scratch=False, train_disc_every=1):

    noise_dim = model.noise_dim
    generator = model.generator
    discriminator = model.discriminator

    number_of_generator_weights = np.sum([np.prod(v.get_shape()) for v in generator.trainable_weights])
    number_of_discrimintor_weights = np.sum([np.prod(v.get_shape()) for v in discriminator.trainable_weights])
    
    # ======================================
    # construct testing data once for all
    # ======================================
    AUTO = tf.data.experimental.AUTOTUNE
    noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], noise_dim))

    test_in = np.concatenate(
        [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise
    cond_dim = test_in.shape[1] - noise_dim
    testing_data = tf.data.Dataset.from_tensor_slices(
        (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)
    
    # ====================================
    # Checkpoints and model summary
    # ====================================
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    checkpoint_dir = os.path.join(log_dir, "checkpoints", f"{time_stamp}")
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
    if not train_from_scratch:
        logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
        _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

    summary_dir = os.path.join(log_dir, "logs", f"{time_stamp}")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    img_dir = os.path.join(log_dir, 'img', f'{time_stamp}')
    os.makedirs(img_dir, exist_ok=True)

    config_path = os.path.join(log_dir, "config.json")
    file_abs_path = []
    for filename in config['filename']:
        file_abs_path.append(os.path.abspath(filename))
    config['filename'] = file_abs_path
    config['log_dir'] = os.path.abspath(config['log_dir'])
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    # ======================
    # Create optimizers
    # ======================
    end_lr = 1e-6
    # gen_lr = keras.optimizers.schedules.PolynomialDecay(gen_lr, max_epochs, end_lr, power=4)
    # disc_lr = keras.optimizers.schedules.PolynomialDecay(disc_lr, max_epochs, end_lr, power=1.0)
    generator_optimizer = keras.optimizers.Adam(gen_lr)
    discriminator_optimizer = keras.optimizers.Adam(disc_lr)

    @tf.function
    def train_step(gen_in_4vec, truth_4vec):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_out_4vec = generator(gen_in_4vec, training=True)

            real_output = discriminator(truth_4vec, training=True)
            fake_output = discriminator(
                tf.concat([gen_out_4vec, gen_in_4vec[:, : cond_dim]], axis=1), 
                training=True
            )

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return disc_loss, gen_loss
    
    @tf.function
    def train_disc_only(gen_in_4vec, truth_4vec):
        condition = tf.cast(gen_in_4vec[:, : cond_dim], tf.float32)
        truth_4vec = tf.cast(truth_4vec, tf.float32)
        with tf.GradientTape() as disc_tape:
            gen_out_4vec = tf.cast(generator(gen_in_4vec, training=True), tf.float32)
            
            real_output = discriminator(
                tf.concat([truth_4vec, condition], axis=1), 
                training=True
            )
            fake_output = discriminator(
                tf.concat([gen_out_4vec, condition], axis=1), 
                training=True
            )
            disc_loss = discriminator_loss(real_output, fake_output)
            gen_loss = generator_loss(fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, gen_loss, gradients_of_discriminator

    @tf.function
    def train_gen_only(gen_in_4vec, truth_4vec):
        condition = tf.cast(gen_in_4vec[:, : cond_dim], tf.float32)
        truth_4vec = tf.cast(truth_4vec, tf.float32)
        with tf.GradientTape() as gen_tape:
            gen_out_4vec = tf.cast(generator(gen_in_4vec, training=True), tf.float32)

            real_output = discriminator(
                tf.concat([truth_4vec, condition], axis=1), 
                training=True
            )
            fake_output = discriminator(
                tf.concat([gen_out_4vec, condition], axis=1), 
                training=True
            )

            disc_loss = discriminator_loss(real_output, fake_output)
            gen_loss = generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        return disc_loss, gen_loss, gradients_of_generator

    @tf.function
    def modified_train_step(gen_in_4vec, truth_4vec, train_disc=True, discriminator_gradient_norm=0, generator_gradient_norm=0):

        if train_disc:
            disc_loss, gen_loss, gradients_of_discriminator = train_disc_only(gen_in_4vec, truth_4vec)
            discriminator_gradient_norm = tf.linalg.global_norm(gradients_of_discriminator) / number_of_discrimintor_weights
        
        disc_loss, gen_loss, gradients_of_generator = train_gen_only(gen_in_4vec, truth_4vec)

        generator_gradient_norm = tf.linalg.global_norm(gradients_of_generator) / number_of_generator_weights

        return disc_loss, gen_loss, discriminator_gradient_norm, generator_gradient_norm

    

    best_wdis = 9999
    best_epoch = -1
    plaeto_threashold = 100000
    num_no_improve = 0
    do_disc_only = False
    num_disc_only_epochs = 0
    i_disc_only = 0
    summary_logfile = os.path.join(summary_dir, f'results_{time_stamp}.txt')
    
    with tqdm.trange(max_epochs, disable=disable_tqdm) as t0:
        for epoch in t0:
            # compose the training dataset by generating different noises for each epochs
            noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], noise_dim))
            train_inputs = np.concatenate(
                [train_in, noise], axis=1).astype(np.float32) if train_in is not None else noise

            dataset = tf.data.Dataset.from_tensor_slices(
                (train_inputs, train_truth)).shuffle(2*batch_size).batch(
                    batch_size, drop_remainder=False).prefetch(AUTO)

            tot_loss = []
            grads = []
            train_disc = (epoch % train_disc_every) == 0
                
            # train_fn = train_disc_only if do_disc_only else train_step
            train_fn = modified_train_step

            i_disc_only += do_disc_only

            for data_batch in dataset:
                [discriminator_gradient_norm, generator_gradient_norm]=grads[-1] if len(grads)>0 else [0,0]
                train_output = train_fn(*data_batch, train_disc=train_disc, discriminator_gradient_norm=discriminator_gradient_norm, generator_gradient_norm=generator_gradient_norm)
                if len(train_output) == 2:
                    tot_loss.append(list(train_output))
                elif len(train_output)==4:
                    tot_loss.append(list(train_output)[:2])
                    grads.append(list(train_output)[2:])
                    

            tot_loss = np.array(tot_loss)
            avg_loss = np.sum(tot_loss, axis=0)/tot_loss.shape[0]
            loss_dict = dict(D_loss=avg_loss[0], G_loss=avg_loss[1])
            grads = np.array(grads)
            print(grads[:,0])
            if grads.shape[0] > 0:
                avg_grads = np.mean(grads, axis=0)
                print(avg_grads[0])
                loss_dict['D_grad_norm']=avg_grads[0]
                loss_dict['G_grad_norm']=avg_grads[1]
            
            predictions, truths = evaluate(generator, testing_data)
            tot_wdis = np.mean([stats.wasserstein_distance(predictions[:, idx], truths[:, idx])\
                    for idx in range(truths.shape[1])])


            with summary_writer.as_default():
                tf.summary.experimental.set_step(epoch)
                tf.summary.scalar("tot_wasserstein_dis",
                    tot_wdis, description="mean wasserstein distance")
                for key,val in loss_dict.items():
                    tf.summary.scalar(key, val)

            if tot_wdis < best_wdis:
                ckpt_manager.save()
                generator.save(os.path.join(log_dir, "generator", f"{time_stamp}"))
                best_wdis = tot_wdis
                best_epoch = epoch
                outname = os.path.join(img_dir, f"{epoch}.png")
                x_ranges = []
                for idx in range( truths.shape[1] ):
                    x_range = [ truths[:, idx].min(), truths[:, idx].max() ]
                    x_ranges.append(x_range)
                compare(predictions, truths, outname, xlabels, x_ranges)

                with open(summary_logfile, 'a') as f:
                    f.write(", ".join(["{:.4f}".format(x) 
                        for x in [best_wdis, best_epoch]]) + '\n')
            else:
                num_no_improve += 1
            
            # if (epoch) % 100 == 0:
            #     outname = os.path.join(img_dir, f"{epoch}.png")
            #     compare(predictions, truths, outname, xlabels)
            
            # so long since last improvement
            # train discriminator only
            if num_no_improve > plaeto_threashold:
                num_no_improve = 0
                num_disc_only_epochs += 1
                do_disc_only = True
            else:
                if i_disc_only == num_disc_only_epochs:
                    do_disc_only = False
                    i_disc_only = 0


            t0.set_postfix(**loss_dict, doOnlyDisc=do_disc_only, BestD=best_wdis, BestE=best_epoch)

    tmp_res = "Best Model in {} Epoch with a Wasserstein distance {:.4f}".format(best_epoch, best_wdis)
    logging.info(tmp_res)
    
    with open(summary_logfile, 'a') as f:
        f.write(tmp_res + "\n")


def inference(gan, test_in, test_truth, log_dir, xlabels):
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(
        generator=gan.generator,
        discriminator=gan.discriminator)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
    logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

    AUTO = tf.data.experimental.AUTOTUNE
    noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], gan.noise_dim))
    test_in = np.concatenate(
        [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise
    testing_data = tf.data.Dataset.from_tensor_slices(
        (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

    summary_dir = os.path.join(log_dir, "logs_inference")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    img_dir = os.path.join(log_dir, 'img_inference')
    os.makedirs(img_dir, exist_ok=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("--model", choices=all_gans, help='gan model', default=None)
    add_arg("--filename", help='input filename', default=None, nargs='+')
    add_arg("--config-file", help='config file name', default=None)
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--num-test-evts", help='number of testing events', default=10000, type=int)
    add_arg("--inference", help='perform inference only', action='store_true')
    add_arg("-v", '--verbose', help='tf logging verbosity', default='INFO',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    add_arg("--gen-lr", help='generator learning rate', type=float, default=0.0001)
    add_arg("--disc-lr", help='discriminator learning rate', type=float, default=0.0001)
    add_arg("--disc-dense-layers", help='discriminator dense layers', type=int, nargs='+', default=[256, 256])
    add_arg("--disc-dropout", help='discriminator learning rate', type=float, default=0.)
    add_arg("--train-from-scratch", help='prevent restoring the best iteration from log dir', action='store_true')
    add_arg("--data", default='herwig_angles',
        choices=['herwig_angles', 'geant4_COM_delta_phi_eta_E_pT', 'geant4_COM_eta_E_pT', 'dimuon_inclusive', 'herwig_angles2', 'geant4_leading_products', 'geant4_momentum_transfer', 'geant4_COM_frame', 'geant4_momentum_transfer_delta_phi_eta', 'geant4_momentum_transfer_delta_phi'])

    # model parameters
    add_arg("--noise-dim", type=int, default=4, help="noise dimension")
    add_arg("--gen-output-dim", type=int, default=2, help='generator output dimension')
    add_arg("--cond-dim", type=int, default=0, help='dimension of conditional input')
    add_arg("--disable-tqdm", action="store_true", help='disable tqdm')

    args = parser.parse_args()

    from tensorflow.compat.v1 import logging
    logging.set_verbosity(args.verbose)

    config = vars(args)
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            for name, value in json.load(f).items():
                config[name] = value

    # prepare input data by calling those function implemented in 
    # gan4hep.preprocess.
    # train_in, train_truth, test_in, test_truth, xlabels = eval(config["data"])(
    #     config["filename"], max_evts=config["max_evts"], print_info=False)
    train_in, train_truth, test_in, test_truth, xlabels = eval(config["data"])(
        config["filename"], max_evts=config["max_evts"], save_transformer=config.get('save_transformer'))

    batch_size = config["batch_size"]
    gan = eval(config["model"])(**config)
    if config["inference"]:
        inference(gan, test_in, test_truth, config["log_dir"], xlabels)
    else:
        train(train_truth, test_truth, gan, config["gen_lr"], config["disc_lr"],
            batch_size, config["epochs"], config["log_dir"], xlabels, config, config["disable_tqdm"],
            train_in, test_in, config["train_from_scratch"], config.get('train_disc_every', 1))
