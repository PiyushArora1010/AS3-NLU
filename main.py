import argparse
from train import trainer
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--model_name', type=str, help='model name', default="facebook/bart")
parser.add_argument('--dataset_tag', type=str, help='dataset tag', default="IndicHeadlineGeneration")
parser.add_argument('--run_name', type=str, help='run name', default="run1")
parser.add_argument('--learning_rate', type=float, help='learning rate', default=5e-5)
parser.add_argument('--batch_size', type=int, help='batch size', default=250)
parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.01)
parser.add_argument('--epochs', type=int, help='epochs', default=50)
parser.add_argument('--log_dir', type=str, help='log dir', default="./logs/")

args = parser.parse_args()

if __name__ == "__main__":
    trainer = trainer(vars(args))
    trainer.run()
    