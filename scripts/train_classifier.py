import argparse
import json

from nnlib.nnlib import training, metrics, callbacks
from nnlib.nnlib.data_utils.wrappers import ResizeImagesWrapper
from nnlib.nnlib.data_utils.base import get_loaders_from_datasets, get_input_shape
from modules.data_utils import load_data_from_arguments
import methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--all_device_ids', nargs='+', type=str, default=None,
                        help="If not None, this list specifies devices for multiple GPU training. "
                             "The first device should match with the main device (args.device).")
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epochs', '-e', type=int, default=400)
    parser.add_argument('--stopping_param', type=int, default=2**30)
    parser.add_argument('--save_iter', '-s', type=int, default=10)
    parser.add_argument('--vis_iter', '-v', type=int, default=10)
    parser.add_argument('--log_dir', '-l', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    # data parameters
    parser.add_argument('--dataset', '-D', type=str, default='corrupt4_mnist')
    parser.add_argument('--data_augmentation', '-A', action='store_true', dest='data_augmentation')
    parser.set_defaults(data_augmentation=False)
    parser.add_argument('--error_prob', '-n', type=float, default=0.0)
    parser.add_argument('--num_train_examples', type=int, default=None)
    parser.add_argument('--clean_validation', action='store_true', default=False)
    parser.add_argument('--resize_to_imagenet', action='store_true', default=False)

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='StandardClassifier')
    parser.add_argument('--load_from', type=str, default=None)

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    args = parser.parse_args()
    print(args)

    # Load data
    train_data, val_data, test_data, _ = load_data_from_arguments(args, build_loaders=False)

    # resize if needed
    if args.resize_to_imagenet:
        train_data = ResizeImagesWrapper(train_data, size=(224, 224))
        val_data = ResizeImagesWrapper(val_data, size=(224, 224))
        test_data = ResizeImagesWrapper(test_data, size=(224, 224))

    train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data=train_data,
                                                                      val_data=val_data,
                                                                      test_data=test_data,
                                                                      batch_size=args.batch_size)

    # Options
    optimization_args = {
        'optimizer': {
            'name': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum
        },
        # 'scheduler': {
        #     'step_size': 15,
        #     'gamma': 1.25
        # }
    }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(methods, args.model_class)

    model = model_class(input_shape=get_input_shape(train_loader.dataset),
                        architecture_args=architecture_args,
                        device=args.device,
                        load_from=args.load_from)

    metrics_list = [metrics.Accuracy(output_key='pred')]
    if args.dataset == 'imagenet':
        metrics_list.append(metrics.TopKAccuracy(k=5, output_key='pred'))

    callbacks_list = [callbacks.SaveBestWithMetric(metric=metrics_list[0], partition='val', direction='max')]

    stopper = callbacks.EarlyStoppingWithMetric(metric=metrics_list[0], stopping_param=args.stopping_param,
                                                partition='val', direction='max')

    training.train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs,
                   save_iter=args.save_iter,
                   vis_iter=args.vis_iter,
                   optimization_args=optimization_args,
                   log_dir=args.log_dir,
                   args_to_log=args,
                   stopper=stopper,
                   metrics=metrics_list,
                   callbacks=callbacks_list,
                   device_ids=args.all_device_ids)


if __name__ == '__main__':
    main()
