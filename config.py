config = {
    # 'train_file_path' : 'Program/data/SemEval-2016-master/train/text.train.conll',
    # 'dev_file_path' : 'Program/data/SemEval-2016-master/validation/text.valid.conll',
    # 'test_file_path' : 'Program/data/SemEval-2016-master/test/text.test.conll',
    # 'glove_file_path' : 'Program/data/glove/glove.6B.100d.txt',
    # 'train_file_path' : './microdata/trail.train.conll',
    # 'dev_file_path' : './microdata/trail.dev.conll',
    # 'test_file_path' : './microdata/trail.test.conll',
    'pretrain_path': './data/giga.100.txt',
    'pretrain_char_path': './data/giga.chars.100.txt',
    # 'glove_file_path' : './microdata/trail.emb',
    'train_file_path' : './SemEval-2016/train/text.train.conll',
    'dev_file_path' : './SemEval-2016/validation/text.valid.conll',
    'test_file_path' : './SemEval-2016/test/text.test.conll',
    'batch_size' : 5000,
    'shuffle' : True,
    'lr' : 2e-3,
    'device' : 'cuda:4',
    'min_freq' : 3,
    'batch_first' : True,
    'out_file_path' : 'Program/SDP/output.txt',
    'model_path' : './models/model_{}.pt',
    'dev_output_path' : "./results/dev/best_dev_results_{}.txt",
    'test_output_path': "./results/test/test_results_{}.txt",
}