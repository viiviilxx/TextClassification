import optuna
import argparse
import torch
from train import Model


parser = argparse.ArgumentParser()
parser.add_argument('--tuning', action='store_true', default=False, help='hyper parameter tuning')
parser.add_argument('--no_cuda', action = 'store_true', default = False)
args = parser.parse_args()


def main() :
    TRAIN_PATH = 'data/id/rcv1-train'
    VAL_PATH = 'data/id/rcv1-val'
    TEST_PATH = 'data/id/rcv1-test'
    embedding_dim = 768

    if args.no_cuda :
        use_cuda = False    
    else :
        use_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.tuning :
        params = {
            'train_path' : TRAIN_PATH,
            'test_path' : TEST_PATH,
            'val_path' : VAL_PATH,
            'tuning' : args.tuning,
            'epoch' : 2,
            'cuda' : use_cuda,
            'classes' : 103,
            'embedding_dim' : embedding_dim,
            'sequence_length' : 512,
            'seed' : 0,
        }

        model = Model(params)
        study = optuna.create_study()
        study.optimize(model.tuning, n_trials = 600)

        print(study.best_trial)

    else :
        params = {
            'train_path' : TRAIN_PATH,
            'test_path' : TEST_PATH,
            'val_path' : VAL_PATH,
            'tuning' : args.tuning,
            'epoch' : 5,
            'cuda' : use_cuda,
            'classes' : 103,
            'embedding_dim' : embedding_dim,
            'sequence_length' : 512,
            'seed' : 0,
        }

        learning_params = {
            'batch_size' : 4,  
            'threshold' : 0.3,
            'patience' : 999999,
            'learning_rate' : 0.0005,
            'warmup_steps' : 1,
            'weight_decay' : 0.00025,
        }

        cnn_params = {  
            'cnn_out_channels' : 512,
            'cnn_filter_sizes' : [3, 9, 27],
            'cnn_hidden_dim1' : 1024,
            'cnn_conv_stride' : 2,
            'cnn_pool_stride' : 8,
            'cnn_dropout' : [False, False],
        }
        
        params.update(learning_params)
        params.update(cnn_params)
        model = Model(params)

        model.run()


if __name__ == "__main__" :
    main()