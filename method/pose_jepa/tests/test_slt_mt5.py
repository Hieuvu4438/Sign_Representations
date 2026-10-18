from copy import deepcopy
from pathlib import Path
import torch
import pytest
from signrepr.training.slt import SLTBridge
from signrepr.training.pretrain import SSLTrainer
from signrepr.training.checkpoints import save_training
from signrepr.training.sequence_tasks import train_sequence
from test_downstream_workflows import supervised_config


def tiny_mt5():
    transformers = pytest.importorskip('transformers')
    cfg = transformers.MT5Config(vocab_size=32, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1,
                                 num_heads=2, d_kv=8, dropout_rate=0., decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    return transformers.MT5ForConditionalGeneration(cfg)


class FixtureTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs['truncation'] is False
        return {'input_ids': torch.tensor([[2, 3, 1]]*len(texts)), 'attention_mask': torch.ones(len(texts), 3, dtype=torch.long)}

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [' '.join(str(int(x)) for x in row if x > 1) for row in sequences]


def test_actual_mt5_bridge_backward_and_generation():
    model = SLTBridge(tiny_mt5(), input_dim=64)
    f, lengths = torch.randn(2, 7, 64), torch.tensor([7, 4])
    labels = torch.tensor([[2, 3, 1], [4, 1, -100]])
    result = model(f, lengths, labels)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert torch.isfinite(model.bridge[0].weight.grad).all()
    assert model.generate(f, lengths, num_beams=2, max_new_tokens=4).shape[0] == 2


def test_complete_slt_tiny_mt5_workflow(config, batch, tmp_path):
    pytest.importorskip('sacrebleu')
    pytest.importorskip('rouge_score')
    ssl = SSLTrainer(config); ssl.update([batch]); save_training(ssl, tmp_path/'ssl.pt')
    cfg = supervised_config(config, tmp_path)
    cfg['downstream'].update(decoder_lr=.0003, target_max_tokens=32, beam=2, max_new_tokens=4, tokenize='13a')
    result = train_sequence(cfg, tmp_path/'ssl.pt', 'slt', decoder=tiny_mt5(), tokenizer=FixtureTokenizer())
    state = torch.load(result['checkpoint'], weights_only=False)
    assert state['validation']['num_utterances'] == 8
    assert 'BLEU_signature' in state['validation']
    assert (Path(cfg['run']['output_dir'])/'train.json').exists()
