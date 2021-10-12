def config():
    configSetting = {
        'dataset_name': 'Dataset',
        'train_name': 'train.name.h5',
        'train_api': 'train.apiseq.h5',
        'train_tokens': 'train.tokens.h5',
        'train_desc': 'train.desc.h5',
        'valid_name': 'test.name.h5',
        'valid_api': 'test.apiseq.h5',
        'valid_tokens': 'test.tokens.h5',
        'valid_desc': 'test.desc.h5',
        'name_len': 6,
        'api_len': 30,
        'tokens_len': 50,
        'desc_len': 30,
        'vocab_size': 10000,
        'vocab_name': 'vocab.name.json',
        'vocab_api': 'vocab.apiseq.json',
        'vocab_tokens': 'vocab.tokens.json',
        'vocab_desc': 'vocab.desc.json',
        'batch_size': 512,
        'Epoch': 15,
        'learning_rate': 1e-5,
        'adam_epsilon': 1e-8,
        'warmup_steps': 5000,
        'fp16': False,
        'd_word_dim': 128,#可以修改
        'd_model': 128,
        'd_ffn':512,
        'n_heads': 16,
        'n_layers': 1,#可以修改
        'd_k': 8,
        'd_v': 8,
        'pad_idx': 0,
        'margin': 0.3986,
        'sim_measure': 'cos'
    }
    return configSetting
