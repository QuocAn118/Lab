"""Notebook support: timing, model export and loss display.
Sources: docs/08_implementation.md §6; docs/04_training.md §6;
docs/05_evaluation.md; docs/11_AI_Usage.md.
"""
from pathlib import Path
from datetime import datetime
import json
import csv
import time
from transformers import TrainerCallback
from reporting import RunReport

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

    def on_log(self, args, state, control, logs=None, **kwargs):
        if getattr(self, 'report', None) is not None:
            self.report.event('trainer_log', step=state.global_step, epoch=state.epoch, values=logs or {})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if getattr(self, 'report', None) is not None:
            self.report.event('validation', step=state.global_step, epoch=state.epoch, values=metrics or {})

    def on_save(self, args, state, control, **kwargs):
        row = {'epoch': state.epoch, 'step': state.global_step,
               'saved_at': datetime.now().astimezone().isoformat(),
               'epoch_seconds': round(time.perf_counter() - self.epoch_start, 3),
               'elapsed_train_seconds': round(time.perf_counter() - self.start, 3)}
        self.epochs.append(row)
        path = Path(args.output_dir)/f'checkpoint-{state.global_step}'/'training_time.json'
        save_training_time(path, {'scope': 'checkpoint', 'started_at': self.started_at, **row})
        if getattr(self, 'report', None) is not None:
            self.report.event('checkpoint_saved', path=str(path.parent), **row)


def trainer_config(trainer, checkpoint=None):
    names = ['learning_rate', 'per_device_train_batch_size', 'per_device_eval_batch_size',
             'num_train_epochs', 'weight_decay', 'max_grad_norm', 'optim', 'seed', 'data_seed',
             'fp16', 'bf16', 'eval_strategy', 'save_strategy', 'save_total_limit',
             'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better']
    return {'checkpoint': str(checkpoint) if checkpoint else None,
            'model_class': type(trainer.model).__name__, 'model_config': trainer.model.config.to_dict(),
            'training_args': {name: getattr(trainer.args, name, None) for name in names},
            'device': str(trainer.args.device),
            'datasets': {name: {'size': len(data), 'fingerprint': getattr(data, '_fingerprint', None)}
                         for name, data in [('train', trainer.train_dataset), ('validation', trainer.eval_dataset)]
                         if data is not None}}


def train_and_save(trainer, tokenizer, output, report_root=None):
    output = Path(output)/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    trainer.args.output_dir = str(output)
    (output/'SOURCES.md').write_text(
        '# Nguồn và phạm vi\n\nÁp dụng mọi artifact trong run này.\n'
        'Căn cứ: docs/08_implementation.md §9; docs/04_training.md §6; docs/05_evaluation.md.\n',
        encoding='utf-8')
    with RunReport('train', root=report_root) as report:
        report.meta['model_output'] = str(output.resolve())
        report.meta['log_scope'] = 'trainer.train() and model export; data/model preparation precedes capture'
        report.write_json('config.json', trainer_config(trainer))
        timing = TrainingTime()
        timing.report = report
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
                       'epochs': getattr(timing, 'epochs', []), 'report_path': str(report.directory.resolve())}
            save_training_time(output/'training_time.json', summary)
            report.write_json('training_time.json', summary)
            report.write_json('trainer_history.json', {'history': trainer.state.log_history})
            report.write_json('checkpoints.json', {'best_model_checkpoint': trainer.state.best_model_checkpoint,
                'checkpoints': [{**row, 'path': str(output/f"checkpoint-{row['step']}"),
                                 'present_at_end': (output/f"checkpoint-{row['step']}").is_dir()}
                                for row in summary['epochs']]})
            trainer.remove_callback(timing)
        checkpoint = output/'best'
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(str(checkpoint))
        trainer.save_state()
        save_training_time(checkpoint/'training_time.json', summary)
        report.meta['saved_model'] = str(checkpoint.resolve())
        print('Model:', checkpoint)
        print('Report:', report.directory)
    return checkpoint


def show_training_time(checkpoint):
    path = Path(checkpoint)/'training_time.json'
    if path.exists():
        value = json.loads(path.read_text(encoding='utf-8'))['total_train_seconds']
        print('Thời gian train đã lưu:', value, 'giây')
    else:
        print('Checkpoint chưa có log thời gian train.')


def plot_loss(trainer, checkpoint, root, training, report=None):
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
    if report is not None:
        report.write_json('loss_history.json', {'available': bool(history), 'history': history})
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
    if report is not None:
        plt.gcf().savefig(report.path('loss_curve.png'), dpi=160, bbox_inches='tight', metadata={'Sources': 'docs/05_evaluation.md'})
    plt.show()
    plt.close()


def infer_and_report(text, report_root=None):
    from transformers import pipeline
    with RunReport('infer', root=report_root) as report:
        report.meta['log_scope'] = 'pretrained model load, tokenization and inference'
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Text must be a non-empty string')
        model_id = 'tabularisai/multilingual-sentiment-analysis'
        report.write_json('config.json', {'model': model_id, 'max_length': 512, 'text': text})
        classifier = pipeline('text-classification', model=model_id)
        tokens = classifier.tokenizer.tokenize(text)
        predictions = classifier(text, truncation=True, max_length=512)
        report.write_json('inference.json', {'model': model_id, 'text': text, 'tokens': tokens,
                          'model_revision': getattr(classifier.model.config, '_commit_hash', None),
                          'predictions': predictions})
        report.event('inference_completed', predictions=predictions)
        print('Tokens:', tokens)
        print(predictions)
        print('Report:', report.directory)
    return predictions


def evaluate_and_report(trainer, dataset, checkpoint, root, training, sentences=None, report_root=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from transformers import pipeline
    sentences = sentences if sentences is not None else [
        'I loved this movie. The acting was great!', 'The movie was boring and disappointing.']
    with RunReport('evaluate', root=report_root) as report:
        report.meta['log_scope'] = 'test prediction, figures and checkpoint demo; model/data preparation precedes capture'
        report.write_json('config.json', {**trainer_config(trainer, checkpoint),
                          'test_dataset': {'size': len(dataset), 'fingerprint': getattr(dataset, '_fingerprint', None)}})
        timing_path = Path(checkpoint)/'training_time.json'
        if timing_path.is_file():
            report.copy(timing_path, 'checkpoint_training_time.json')
        start = time.perf_counter()
        result = trainer.predict(dataset)
        report.write_json('evaluation_time.json', {'predict_seconds': time.perf_counter()-start})
        report.write_json('metrics.json', result.metrics)
        predicted = np.argmax(result.predictions, axis=-1)
        with report.path('predictions.csv').open('w', encoding='utf-8', newline='') as stream:
            stream.write('# Sources: docs/05_evaluation.md; docs/08_implementation.md §9\n')
            writer = csv.writer(stream)
            writer.writerow(['sample_index', 'actual', 'predicted'])
            writer.writerows(zip(range(len(predicted)), result.label_ids.tolist(), predicted.tolist()))
        matrix = confusion_matrix(result.label_ids, predicted, labels=[0, 1])
        report.write_json('confusion_matrix.json', {'labels': ['NEGATIVE', 'POSITIVE'], 'matrix': matrix})
        display = ConfusionMatrixDisplay.from_predictions(result.label_ids, predicted,
                  labels=[0, 1], display_labels=['NEGATIVE', 'POSITIVE'], cmap='Blues')
        display.figure_.savefig(report.path('confusion_matrix.png'), dpi=160, bbox_inches='tight',
                               metadata={'Sources': 'docs/05_evaluation.md'})
        plt.show()
        plt.close(display.figure_)
        print(result.metrics)
        report.event('test_evaluated', metrics=result.metrics, samples=len(predicted))
        plot_loss(trainer, checkpoint, root, training, report)
        import torch
        sentiment = pipeline('text-classification', model=str(checkpoint), tokenizer=str(checkpoint),
                             device=0 if torch.cuda.is_available() else -1)
        predictions = sentiment(sentences, truncation=True, max_length=256)
        demo = [{'text': text, **prediction} for text, prediction in zip(sentences, predictions)]
        report.write_json('demo.json', {'checkpoint': str(checkpoint), 'predictions': demo})
        for row in demo:
            print(row['text'], '->', {'label': row['label'], 'score': row['score']})
        print('Report:', report.directory)
    return result
