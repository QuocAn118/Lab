"""Notebook support: timing, model export and loss display.
Sources: docs/08_implementation.md §6; docs/04_training.md §6;
docs/05_evaluation.md; docs/11_AI_Usage.md.
"""
from pathlib import Path
from datetime import datetime
import json
import time
from transformers import TrainerCallback

def save_training_time(path, values):
    values = {'sources_md': ['docs/04_training.md', 'docs/08_implementation.md'], **values}
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')

class TrainingTime(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = datetime.now().astimezone().isoformat()
        self.start = time.perf_counter()
        self.epochs = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.perf_counter()

    def on_save(self, args, state, control, **kwargs):
        row = {'epoch': state.epoch, 'step': state.global_step,
               'saved_at': datetime.now().astimezone().isoformat(),
               'epoch_seconds': round(time.perf_counter() - self.epoch_start, 3),
               'elapsed_train_seconds': round(time.perf_counter() - self.start, 3)}
        self.epochs.append(row)
        path = Path(args.output_dir)/f'checkpoint-{state.global_step}'/'training_time.json'
        save_training_time(path, {'scope': 'checkpoint', 'started_at': self.started_at, **row})


def train_and_save(trainer, tokenizer, output):
    output = Path(output)/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    trainer.args.output_dir = str(output)
    (output/'SOURCES.md').write_text(
        '# Nguồn và phạm vi\n\nÁp dụng mọi artifact trong run này.\n'
        'Căn cứ (từ Lab 3/): docs/08_implementation.md §6; docs/04_training.md §6; docs/03_model.md; docs/02_data.md.\n',
        encoding='utf-8')
    timing = TrainingTime()
    trainer.add_callback(timing)
    started_at, start = datetime.now().astimezone().isoformat(), time.perf_counter()
    status = 'interrupted_or_failed'
    try:
        trainer.train()
        status = 'completed'
    finally:
        summary = {'scope': 'run', 'status': status,
                   'started_at': getattr(timing, 'started_at', started_at),
                   'ended_at': datetime.now().astimezone().isoformat(),
                   'total_train_seconds': round(time.perf_counter() - getattr(timing, 'start', start), 3),
                   'best_model_checkpoint': trainer.state.best_model_checkpoint,
                   'epochs': getattr(timing, 'epochs', [])}
        save_training_time(output/'training_time.json', summary)
        trainer.remove_callback(timing)
    checkpoint = output/'best'
    trainer.save_model(str(checkpoint))
    tokenizer.save_pretrained(str(checkpoint))
    trainer.save_state()
    save_training_time(checkpoint/'training_time.json', summary)
    return checkpoint


def show_training_time(checkpoint):
    path = Path(checkpoint)/'training_time.json'
    if path.exists():
        value = json.loads(path.read_text(encoding='utf-8'))['total_train_seconds']
        print('Thời gian train đã lưu:', value, 'giây')
    else:
        print('Checkpoint chưa có log thời gian train.')


def plot_loss(trainer, checkpoint, root, training):
    import matplotlib.pyplot as plt
    history_path = Path(checkpoint).parent/'trainer_state.json'
    full_history_path = Path(root)/'outputs/full/trainer_history.json'
    if training:
        history = trainer.state.log_history
    elif history_path.exists():
        history = json.loads(history_path.read_text(encoding='utf-8'))['log_history']
    elif full_history_path.exists():
        history = json.loads(full_history_path.read_text(encoding='utf-8'))['history']
    else:
        history = []
    if not history:
        print('Không có log loss của lượt train cũ để vẽ lại biểu đồ.')
        return
    train_log = [row for row in history if 'loss' in row]
    val_log = [row for row in history if 'eval_loss' in row]
    plt.plot([row['step'] for row in train_log], [row['loss'] for row in train_log], label='Train')
    plt.plot([row['step'] for row in val_log], [row['eval_loss'] for row in val_log], 'o-', label='Validation')
    plt.xlabel('Optimizer step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
