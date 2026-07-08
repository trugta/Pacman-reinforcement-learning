import tensorflow as tf

print('TensorFlow version:', tf.__version__)
GPUs = tf.config.list_physical_devices('GPU')
print('GPUs found:', GPUs)
if GPUs:
    for g in GPUs:
        try:
            tf.config.experimental.set_memory_growth(g, True)
            print('Enabled memory growth for', g)
        except Exception as e:
            print('Could not set memory growth for', g, 'error:', e)
else:
    print('No GPUs available to enable memory growth.')
