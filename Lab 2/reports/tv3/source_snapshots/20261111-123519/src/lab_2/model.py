"""Adapter package cho model.py của Xuân Bảo; Codex tích hợp 2026-11-11.

Giữ nguyên implementation gốc, chỉ xuất API cho pipeline dùng chung.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_spec = spec_from_file_location("lab2_existing_model", Path(__file__).resolve().parents[2] / "model.py")
_source = module_from_spec(_spec)
_spec.loader.exec_module(_source)

NUM_CLASSES = _source.NUM_CLASSES
HEADS = _source.HEADS
SUPPORTED_MODELS = _source.SUPPORTED_MODELS
head_of = _source.head_of
build_model = _source.build_model
trainable_parameters = _source.trainable_parameters
count_parameters = _source.count_parameters
set_train_mode = _source.set_train_mode
save_checkpoint = _source.save_checkpoint
load_checkpoint = _source.load_checkpoint


def main():
    from lab_2.common import argument_parser, load_config, update_report
    config = load_config(argument_parser(__doc__).parse_args().config)
    report = {}
    for label, name, unfreeze in (("resnet18", "resnet18", ()),
                                   ("densenet121", "densenet121", ()),
                                   ("resnet18_unfreeze_layer4", "resnet18", ("layer4",))):
        model = build_model(name, config["model"]["dropout"], unfreeze=unfreeze)
        print(model)
        for key, parameter in model.named_parameters():
            print(key, tuple(parameter.shape), "train" if parameter.requires_grad else "frozen")
        report[label] = count_parameters(model)
    update_report(config, "model", report)


if __name__ == "__main__":
    main()
