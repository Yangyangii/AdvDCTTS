
class ConfigArgs:
    model = 'Text2Mel' # Text2Mel, SSRN
    # speaker = 'kss'
    data_path = '/home/yangyangii/data/LJSpeech-1.1'
    mel_dir, mag_dir = 'd_mels', 'd_mags'
    ga_dir = 'guides' # guided attention
    meta = 'metadata.csv'
    meta_train = 'meta-train.csv'
    meta_eval = 'meta-eval.csv'
    testset = 'test_sents.txt'
    logdir = 'logs'
    sampledir = 'samples'
    testdir = 'tests'
    prepro = True
    mem_mode= True
    ga_mode = True
    log_mode = True
    save_term = 1000
    n_workers = 8
    n_gpu = 1
    global_step = 0

    sr = 22050 # sampling rate
    n_fft = 1024
    n_mags = n_fft//2 + 1
    n_mels = 80
    hop_length = 256
    win_length = 1024
    gl_iter = 100 # Griffin-Lim iteration
    max_db = 100
    ref_db = 20
    power = 1.2
    r = 4  # reduction factor. mel/4
    g = 0.2

    batch_size = 16
    test_batch = 4 # for test
    max_step = 400000
    lr = 0.0001
    lr_decay_step = 50000 # actually not decayed per this step
    Ce = 128  # for text embedding and encoding
    Cx = 256 # for text embedding and encoding
    Cy = 256 # for audio encoding
    Cs = 512 # for SSRN
    drop_rate = 0.05

    max_Tx = 188
    max_Ty = 250

    vocab = u'''PE !',-.?abcdefghijklmnopqrstuvwxyz'''
