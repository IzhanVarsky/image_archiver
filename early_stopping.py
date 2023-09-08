import torch


class EarlyStopping:
    def __init__(self, model_name, patience=15, min_delta=0,
                 save_best=False, use_early_stop=True, metric_decreasing=True):
        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.use_early_stop = use_early_stop
        self.save_best = save_best
        if metric_decreasing:
            self.is_cur_metric_better = lambda val: self.best_metric - val > self.min_delta
        else:
            self.is_cur_metric_better = lambda val: self.best_metric - val < self.min_delta

    def __call__(self, cur_metric, model, out_dir='.'):
        if self.best_metric is None:
            self.best_metric = cur_metric
            if self.save_best:
                self.save_best_model(model, out_dir)
        elif self.is_cur_metric_better(cur_metric):
            self.best_metric = cur_metric
            self.counter = 0
            if self.save_best:
                self.save_best_model(model, out_dir)
        else:
            self.counter += 1
            if self.use_early_stop and self.counter >= self.patience:
                self.early_stop = True

    def save_best_model(self, model, out_dir):
        print("-" * 100)
        print(f">>> Saving the current {self.model_name} model with the best metric value...")
        torch.save(model.state_dict(), f'{out_dir}/{self.get_checkpoint_name()}')

    def get_checkpoint_name(self):
        return f'{self.model_name}_best_metric_model.pth'
