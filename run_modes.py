import os
import time
import torch
import numpy as np
from tqdm import tqdm
from utils import split_list
from model import *
from torch.utils.data import DataLoader
import torch.nn as nn
from cache import *
from codecarbon import EmissionsTracker

# Federated learning requires Flower framework
try:
    import flwr as fl
except ImportError:
    fl = None


def single_cache_test(test_trace, all_pred, save_name, dicts):
    bo_map, bo_map_div, operation_id_map, operation_id_map_div = dicts
    hit_rate = []
    prehit_rate = []
    stats = []
    caches = {}
    maxsize = [5] + \
        [i*10 for i in range(1, 10)] + [i*100 for i in range(1, 11)]

    for i in range(len(maxsize)):
        caches["LRU"+str(maxsize[i])] = CacheTest(maxsize[i])

    for i in tqdm(range(0, len(test_trace))):
        for name, cache in caches.items():
            cache.push_normal(test_trace[i])
            if all_pred[i][0] > 0:
                cache.push_prefetch(
                    test_trace[i] - operation_id_map_div[bo_map_div[all_pred[i][0]-1]])###

    for name, cache in caches.items():
        print(format(cache.get_hit_rate(), '.4f'), format(
            cache.get_prehit_rate(), '.4f'), '\t', name)
        hit_rate.append(cache.get_hit_rate())
        prehit_rate.append(cache.get_prehit_rate())
        stats.append(cache.get_stats())

    np.savetxt('dataset/hit_results/'+save_name+'_hit_rate.txt', hit_rate, fmt='%.4f')
    np.savetxt('dataset/hit_results/'+save_name +
               '_pre_hit_rate.txt', prehit_rate, fmt='%.4f')
    np.savetxt('dataset/hit_results/'+save_name+'_stats.txt', stats, fmt='%d')
    return 0

def score_compute(all_preds,all_targets,save_name):
    
    pre_list = []
    mmr_list = []
    for i in range(1,len(all_preds[0])):
        pre_list.append(np.mean([np.where(t in p[:i],1,0) for t,p in zip(all_targets,all_preds)]))
        # mmr_list.append(np.mean([1/(np.where(p[:i]==t)[0]+1) if t in p[:i] else 0 for t,p in zip(all_targets,all_preds)]))
        mmr_list.append(np.mean([1/(np.where(p[:i]==t)[0][0]+1) if t in p[:i] else 0 for t,p in zip(all_targets,all_preds)]))
    np.savetxt('dataset/hit_results/'+save_name+'_pre_list.txt', pre_list, fmt='%.4f')  
    np.savetxt('dataset/hit_results/'+save_name +'_mmr_list.txt', mmr_list, fmt='%.4f')
    return pre_list,mmr_list


def run_distributed(opt, dataset_col, dataset2input_cached):
    """
    Run distributed learning mode.
    """
    print("Running in Distributed Learning Mode...")
    for dataset in dataset_col:
        train_data_list, train_silces, test_data_list, test_silces, dicts, n_node, train_trace, test_trace = dataset2input_cached(
            dataset=dataset, window_size=opt.window, top_num=opt.topnum)

        # Split data among clients
        num_clients = opt.NUM_CLIENTS
        client_train_data = split_list(train_data_list, num_clients)
        client_test_data = split_list(test_data_list, num_clients)
        client_train_slices = split_list(train_silces, num_clients)  # Split train_silces
        client_test_slices = split_list(test_silces, num_clients)   

        # Initialize models for each client
        client_models = [trans_to_cuda(SessionGraph(opt, n_node)) for _ in range(num_clients)]

        # Create output directories
        model_path = 'checkpoint/' + 'model_' + str(dataset) + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        os.makedirs(model_path, exist_ok=True)
        print('\nModel_Path:', model_path)
        print('\n-------------------------------------------------------\n')

        # Train each client
        for client_id in range(num_clients):
            print(f'\nTraining Client {client_id + 1}/{num_clients}')
            model = client_models[client_id]
            client_train = client_train_data[client_id]
            client_test = client_test_data[client_id]
            client_train_slices = client_train_slices[client_id]  # Use client-specific slices
            client_test_slices = client_test_slices[client_id]

            # Initialize CodeCarbon tracker
            # tracker = EmissionsTracker()
            # tracker.start()
            # start_time = time.time()            

            for epoch in range(opt.epoch):
                print('\n-------------------------------------------------------\n')
                print(f'Client {client_id + 1} | Epoch: {epoch}')
                print('Start training: ')
                all_pred, all_targets = train_test_pred(
                    model, client_train, client_train_slices, client_test, client_test_slices,client_id)

                save_name = dataset+'_'+str(epoch)+'_epoch'
                print('Start cache test: ')
                _ = single_cache_test(
                    test_trace=test_trace[opt.window:-1], all_pred=all_pred, save_name=save_name, dicts=dicts)
                pre, mmr = score_compute(all_preds=all_pred, all_targets=all_targets, save_name=save_name)
                print('Precision:', pre)
                print('MMR:', mmr)

                # Save the model for the client
                torch.save(model, os.path.join(model_path, f'client_{client_id + 1}_epoch_{epoch}.pt'))
            
            # energy_measurement.end()
            # end_time = time.time()
            # total_time = end_time - start_time
            # emissions = tracker.stop()

            # print(f"Dsitributed Learning Completed. Total Time Taken: {total_time:.2f} seconds.")
            # print(f"Carbon Emissions: {emissions} kg CO2")

        # plot_overall_client_predictions(client_logs_across_epochs)
        torch.cuda.empty_cache()
        print('-------------------------------------------------------')


def run_federated(opt, dataset_col, dataset2input_cached):
    """
    Run federated learning mode using Flower.
    """
    if fl is None:
        raise ImportError("Flower is not installed. Please install flower via `pip install flwr` to use federated learning.")

    print("Running in Federated Learning Mode...")# Optional if mixing TF/PyTorch or running in GPU contexts

    for dataset in dataset_col:
        # Load and cache dataset
        train_data_list, train_slices, test_data_list, test_slices, dicts, n_node, train_trace, test_trace = dataset2input_cached(
            dataset=dataset, window_size=opt.window, top_num=opt.topnum)

        # Split dataset across clients
        num_clients = opt.NUM_CLIENTS
        client_train_data = split_list(train_data_list, num_clients)
        client_test_data = split_list(test_data_list, num_clients)
        client_train_slices = split_list(train_slices, num_clients)
        client_test_slices = split_list(test_slices, num_clients)

        # ===== Flower Client Definition =====
        class FederatedClient(fl.client.NumPyClient):
            def __init__(self, client_id, model, train_data, train_slices, test_data, test_slices):
                self.client_id = client_id
                self.model = model
                self.train_data = train_data
                self.train_slices = train_slices
                self.test_data = test_data
                self.test_slices = test_slices

            def get_parameters(self, config=None):
                return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

            def set_parameters(self, parameters):
                state_dict = self.model.state_dict()
                for k, v in zip(state_dict.keys(), parameters):
                    state_dict[k] = torch.tensor(v)
                self.model.load_state_dict(state_dict)

            def fit(self, parameters, config=None):
                self.set_parameters(parameters)
                print(f"Client {self.client_id} training...")
                train_test_pred(
                    self.model,
                    self.train_data,
                    self.train_slices,
                    self.test_data,
                    self.test_slices,
                    self.client_id,
                )
                return self.get_parameters(), len(self.train_slices), {}

            def evaluate(self, parameters, config=None):
                self.set_parameters(parameters)
                print(f"Client {self.client_id} evaluating...")
                all_pred, all_targets = train_test_pred(
                    self.model,
                    self.train_data,
                    self.train_slices,
                    self.test_data,
                    self.test_slices,
                    self.client_id,
                )
                accuracy = np.mean([
                    1 if target in pred[:opt.topn] else 0
                    for target, pred in zip(all_targets, all_pred)
                ])
                return float(0.0), len(self.test_slices), {"accuracy": float(accuracy)}

        # ===== Start Federated Simulation =====
        def client_fn(cid: str):
            client_id = int(cid)
            model = trans_to_cuda(SessionGraph(opt, n_node))
            return FederatedClient(
                client_id,
                model,
                client_train_data[client_id],
                client_train_slices[client_id],
                client_test_data[client_id],
                client_test_slices[client_id],
            )

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
        )

        print("\nStarting federated training with Flower...\n")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=opt.epoch),
            strategy=strategy,
        )

        print("\nFederated Learning Completed.")
        torch.cuda.empty_cache()


# def run_federated(opt, dataset_col, dataset2input_cached):
#     """
#     Manually run federated learning using FedAvg (no Flower).
#     """
#     print("Running Manual Federated Learning with FedAvg...")
    
#     for dataset in dataset_col:
#         print(f"Loading dataset: {dataset}")
#         train_data_list, train_slices, test_data_list, test_slices, dicts, n_node, train_trace, test_trace = dataset2input_cached(
#             dataset=dataset, window_size=opt.window, top_num=opt.topnum)

#         num_clients = opt.NUM_CLIENTS
#         client_train_data = split_list(train_data_list, num_clients)
#         client_test_data = split_list(test_data_list, num_clients)

#         # Initialize global model
#         global_model = trans_to_cuda(SessionGraph(opt, n_node))

#         # Create output directories
#         model_path = 'checkpoint/' + 'fedavg_' + str(dataset) + '_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
#         os.makedirs(model_path, exist_ok=True)
#         print(f"Model path: {model_path}")

#         # Initialize models for each client
#         client_models = [trans_to_cuda(SessionGraph(opt, n_node)) for _ in range(num_clients)]

#         for round in range(opt.epoch):
#             print(f"\n=================== Federated Round {round+1}/{opt.epoch} ===================")

#             # Step 1: Send global model to all clients
#             global_params = global_model.state_dict()
#             for model in client_models:
#                 model.load_state_dict(global_params)

#             local_params = []
#             local_data_sizes = []

#             # Step 2: Train each client locally
#             for client_id in range(num_clients):
#                 print(f"\n[Client {client_id}] Local training...")
#                 model = client_models[client_id]
#                 train_data = client_train_data[client_id]
#                 train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True)
#                 optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
#                 criterion = nn.CrossEntropyLoss()

#                 model.train()
#                 for batch in tqdm(train_loader, desc=f"Client {client_id}"):
#                     alias_inputs, A, items, mask, targets = batch
#                     optimizer.zero_grad()
#                     targets = trans_to_cuda(torch.Tensor(targets).long())
#                     scores = model(items, A)
#                     loss = criterion(scores, targets)
#                     loss.backward()
#                     optimizer.step()

#                 # Save model parameters and size for aggregation
#                 local_params.append({k: v.cpu().detach().clone() for k, v in model.state_dict().items()})
#                 local_data_sizes.append(len(train_data))

#             # Step 3: FedAvg aggregation
#             print("\nAggregating client models with FedAvg...")
#             new_global_params = {}
#             total_data = sum(local_data_sizes)

#             for k in global_params.keys():
#                 new_global_params[k] = sum(
#                     (local_params[i][k] * (local_data_sizes[i] / total_data) for i in range(num_clients))
#                 )

#             global_model.load_state_dict(new_global_params)

#             # Optional: evaluate global model
#             global_model.eval()
#             test_loader = DataLoader(sum(client_test_data, []), batch_size=opt.batchSize, shuffle=False)
#             total, correct, loss = 0, 0, 0
#             criterion = nn.CrossEntropyLoss()

#             with torch.no_grad():
#                 for batch in test_loader:
#                     alias_inputs, A, items, mask, targets = batch
#                     targets = trans_to_cuda(torch.Tensor(targets).long())
#                     scores = global_model(items, A)
#                     loss += criterion(scores, targets).item()
#                     _, predicted = scores.max(1)
#                     total += targets.size(0)
#                     correct += predicted.eq(targets).sum().item()

#             acc = correct / total
#             print(f"[Round {round+1}] Eval Loss: {loss:.4f}, Accuracy: {acc:.4f}")

#         # Final save
#         torch.save(global_model.state_dict(), os.path.join(model_path, f"global_model_final.pt"))
#         print(f"Saved final global model to: {model_path}")
