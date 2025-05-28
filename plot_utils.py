import matplotlib.pyplot as plt
import os

def plot_overall_client_predictions(client_logs_across_epochs, save_dir='client_plots'):
    os.makedirs(save_dir, exist_ok=True)

    for client_id, epoch_logs in client_logs_across_epochs.items():
        plt.figure(figsize=(12, 6))
        plt.title(f'Client {client_id} Predictions Across Epochs')

        for log in epoch_logs:
            epoch = log['epoch']
            train_pred = [p[0] if len(p) > 0 else -1 for p in log['train_preds']]
            test_pred = [p[0] if len(p) > 0 else -1 for p in log['test_preds']]
            train_true = log['train_targets']
            test_true = log['test_targets']

            x_train = range(len(train_true))
            x_test = range(len(test_true))

            plt.plot(x_train, train_true, label=f'Train Target (Epoch {epoch})', linestyle='--', alpha=0.3)
            plt.plot(x_train, train_pred, label=f'Train Pred (Epoch {epoch})')

            plt.plot(x_test, test_true, label=f'Test Target (Epoch {epoch})', linestyle='--', alpha=0.3)
            plt.plot(x_test, test_pred, label=f'Test Pred (Epoch {epoch})')

        plt.xlabel("Sequence Index")
        plt.ylabel("Item ID")
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'client_{client_id}_prediction_plot.png'))
        plt.close()
