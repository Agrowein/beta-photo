__all__ = ['Pix2PixGAN', 'losses']

import tensorflow as _tf
import numpy as _np
import losses



class Pix2PixGAN:
    def __init__(self, generator:_tf.keras.Model, discriminator:_tf.keras.Model,
                 generator_optimizer:_tf.keras.optimizers.Optimizer, discriminator_optimizer:_tf.keras.optimizers.Optimizer,
                 loss:losses.Pix2PixLoss=losses.Pix2PixLoss()):
        
        self.generator = generator
        self.discriminator = discriminator

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.loss = loss


    
    def fit(self, x_generator, y_generator, iterations):
        @_tf.function
        def train_step(input_image, target):
            with _tf.GradientTape() as gen_tape, _tf.GradientTape() as disc_tape:
                generated = self.generator(input_image, training=True)

                disc_real = self.discriminator(target, training=True)
                disc_fake = self.discriminator(generated, training=True)

                adv_loss, l1_loss, gen_loss = self.loss.generator_loss(disc_fake, generated, target)
                disc_loss = self.loss.discriminator_loss(disc_fake, disc_real)

            gen_gradient = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))

            return adv_loss, l1_loss, gen_loss, disc_loss
        
        for i in range(iterations):
            x = next(x_generator)
            y = next(y_generator)

            adv_loss, l1_loss, gen_loss, disc_loss = train_step(x, y)

            if i % 100 == 0:
                print(f'gen_loss: {gen_loss}, disc_loss: {disc_loss}')