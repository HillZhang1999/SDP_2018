# SDP_2018

### Usage

> usage: python train.py [-h] [--train_dataset TRAIN_DATASET] [--validation_dataset VALIDATION_DATASET] [--test_dataset TEST_DATASET] [--num_workers NUM_WORKERS] [--lr LR] [--save_folder SAVE_FOLDER] [--ngpu NGPU] [--batch_size BATCH_SIZE]
>                 [--epoch EPOCH] [--gpu_train GPU_TRAIN] [--shuffle SHUFFLE] [--word_dim WORD_DIM] [--word2vec_path WORD2VEC_PATH] [--char_dim CHAR_DIM] [--char2vec_path CHAR2VEC_PATH] [--pos_tag_dim POS_TAG_DIM]
>                 [--action_dim ACTION_DIM] [--hidden_dim HIDDEN_DIM] [--num_layers NUM_LAYERS] [--patience PATIENCE] [--model_folder MODEL_FOLDER] [--result_folder RESULT_FOLDER] [--validation_metric VALIDATION_METRIC]

### arguments

>   -h, --help            show this help message and exit
>   --train_dataset TRAIN_DATASET
>                         Training dataset directory
>   --validation_dataset VALIDATION_DATASET
>                         Validation dataset directory
>   --test_dataset TEST_DATASET
>                         Test dataset directory
>   --num_workers NUM_WORKERS
>                         Number of workers used in dataloading
>   --lr LR, --learning-rate LR
>                         Learning rate
>   --save_folder SAVE_FOLDER
>                         Location to save checkpoint models
>   --ngpu NGPU           gpu num for training
>   --batch_size BATCH_SIZE
>                         batch size for training
>   --epoch EPOCH         total epochs for training
>   --gpu_train GPU_TRAIN
>                         whether use gpu for training
>   --shuffle SHUFFLE     whether shuffle the train-set
>   --word_dim WORD_DIM   word_embedding_dim
>   --word2vec_path WORD2VEC_PATH
>                         word2vec_pretrained_file_path
>   --char_dim CHAR_DIM   char_embedding_dim
>   --char2vec_path CHAR2VEC_PATH
>                         char2vec_pretrained_file_path
>   --pos_tag_dim POS_TAG_DIM
>                         pos_tag_embedding_dim
>   --action_dim ACTION_DIM
>                         action_embedding_dim
>   --hidden_dim HIDDEN_DIM
>                         hidden_dim
>   --num_layers NUM_LAYERS
>                         The number of hidden layers
>   --patience PATIENCE   The training is stopped after `patience` epochs with no improvement
>   --model_folder MODEL_FOLDER
>                         the path to save models
>   --result_folder RESULT_FOLDER
>                         the path to save results that our model predicted on validation set
>   --validation_metric VALIDATION_METRIC
>                         Validation metric to measure for whether to stop training using patience and whether to serialize an `is_best` model each epoch. The metric name must be prepended with either "+" or "-", which specifies whether the metric is an increasing or decreasing function