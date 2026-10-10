"""Fairseq --user-dir registration; original architecture except feature input width."""
from fairseq.models import register_model
from fairseq.models.sign_to_text.sign2text_transformer import Sign2TextTransformerModel, Sign2TextTransformerConfig


@register_model("signrep_sign2text_transformer", dataclass=Sign2TextTransformerConfig)
class SignRepSign2TextTransformer(Sign2TextTransformerModel):
    @classmethod
    def build_encoder(cls, cfg, feats_type, feat_dim):
        return super().build_encoder(cfg, feats_type, 768)
