import torch
import torch.nn as nn
import torch.nn.functional as F

from .rsu import RSU7, RSU6, RSU5, RSU4, RSU4F
from .csif import CSIF
from .ssie import SSIE
from .blocks import ConvBNReLU
from .enhancements import (
    DynamicFusionGate,
    ECABlock,
    EnhancementClassifier,
    InputRepresentationEnhancer,
    LearnableGeMPool2d,
    PrototypeRelationRefiner,
    ScaleAttentionFusion,
)


class AGSENetClassifier(nn.Module):
    """
    AGSENet inspired classification model with additive multi-scale and
    prototype-aware enhancements on top of the original RSU + CSIF + SSIE path.

    The original backbone, neck, classifier head, and description-aligned
    auxiliary branch are retained. Extra modules enrich representation quality
    through shallow input enhancement, channel attention, token attention across
    scales, GeM pooling, and dynamic image/text fusion.
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=5,
        base_ch=64,
        dropout=0.3,
        description_embeddings=None,
        description_mix_weight=0.35,
        description_hidden_dim=512,
        description_logit_scale_init=14.2857,
        scale_attention_heads=4,
        fusion_gate_hidden_dim=256,
    ):
        super(AGSENetClassifier, self).__init__()

        self.base_ch = int(base_ch)
        self.input_enhancer = InputRepresentationEnhancer(in_ch)

        # Encoder Stages (U2Net style)
        self.en_1 = RSU7(in_ch, base_ch // 2, base_ch)
        self.en_2 = RSU6(base_ch, base_ch // 2, base_ch)
        self.en_3 = RSU5(base_ch, base_ch // 2, base_ch)
        self.en_4 = RSU4(base_ch, base_ch // 2, base_ch)
        self.en_5 = RSU4F(base_ch, base_ch // 2, base_ch)
        self.en_6 = RSU4F(base_ch, base_ch // 2, base_ch)
        
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # CSIF Modules (Saliency Focus)
        self.csif_1 = CSIF(base_ch)
        self.csif_2 = CSIF(base_ch)
        self.csif_3 = CSIF(base_ch)
        self.csif_4 = CSIF(base_ch)
        self.csif_5 = CSIF(base_ch)
        self.csif_6 = CSIF(base_ch)

        # Additive channel recalibration after each encoder stage.
        self.encoder_eca_1 = ECABlock(base_ch)
        self.encoder_eca_2 = ECABlock(base_ch)
        self.encoder_eca_3 = ECABlock(base_ch)
        self.encoder_eca_4 = ECABlock(base_ch)
        self.encoder_eca_5 = ECABlock(base_ch)
        self.encoder_eca_6 = ECABlock(base_ch)
        
        # Neck Projection to standardize channels before fusion
        self.proj_6 = ConvBNReLU(base_ch, base_ch)
        self.proj_5 = ConvBNReLU(base_ch, base_ch)
        self.proj_4 = ConvBNReLU(base_ch, base_ch)
        self.proj_3 = ConvBNReLU(base_ch, base_ch)
        
        # SSIE Fusion Neck (Spatial Saliency Information Exploration)
        self.ssie_5 = SSIE(base_ch, base_ch, base_ch)
        self.ssie_4 = SSIE(base_ch, base_ch, base_ch)
        self.ssie_3 = SSIE(base_ch, base_ch, base_ch)

        # Decoder-side recalibration and richer pooling.
        self.decoder_eca_6 = ECABlock(base_ch)
        self.decoder_eca_5 = ECABlock(base_ch)
        self.decoder_eca_4 = ECABlock(base_ch)
        self.decoder_eca_3 = ECABlock(base_ch)
        self.gem_pool = LearnableGeMPool2d()
        self.scale_attention = ScaleAttentionFusion(
            token_dim=base_ch,
            num_heads=scale_attention_heads,
            dropout=dropout,
        )

        # Classification Head
        # Global Avg + Max pooling across levels 3, 4, 5, 6
        # Total pooled channels: 4 levels * base_ch * 2 (avg+max) = 8 * base_ch
        in_features = base_ch * 8
        self.pooled_feature_dim = in_features
        self.enhanced_feature_dim = (base_ch * 8) + base_ch + (base_ch * 4)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, out_ch)
        )
        self.enhanced_classifier = EnhancementClassifier(
            in_features=self.enhanced_feature_dim,
            hidden_features=max(in_features, self.enhanced_feature_dim // 2),
            out_ch=out_ch,
            dropout=dropout,
        )
        self.visual_residual_scale = nn.Parameter(torch.tensor(0.5))

        self.use_description_aux = description_embeddings is not None
        self.description_mix_weight = float(description_mix_weight)
        if self.use_description_aux:
            description_embeddings = torch.as_tensor(description_embeddings, dtype=torch.float32)
            if description_embeddings.ndim != 2 or description_embeddings.shape[0] != out_ch:
                raise ValueError(
                    "description_embeddings must be a [num_classes, feature_dim] tensor "
                    f"but got shape {tuple(description_embeddings.shape)}"
                )

            description_dim = int(description_embeddings.shape[1])
            self.register_buffer("description_embeddings", description_embeddings)
            self.description_adapter = nn.Sequential(
                nn.LayerNorm(description_dim),
                nn.Linear(description_dim, description_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(description_hidden_dim, description_dim),
            )
            self.prototype_refiner = PrototypeRelationRefiner(
                feature_dim=description_dim,
                hidden_dim=max(description_hidden_dim, description_dim),
                dropout=dropout,
            )
            self.image_projection = nn.Sequential(
                nn.LayerNorm(self.enhanced_feature_dim),
                nn.Linear(self.enhanced_feature_dim, description_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(description_hidden_dim, description_dim),
            )
            self.dynamic_fusion_gate = DynamicFusionGate(
                visual_dim=self.enhanced_feature_dim,
                text_dim=description_dim,
                logit_dim=out_ch,
                hidden_dim=max(int(fusion_gate_hidden_dim), 64),
                dropout=dropout,
            )
            self.logit_scale = nn.Parameter(
                torch.log(torch.tensor(float(description_logit_scale_init)))
            )
        else:
            self.register_buffer("description_embeddings", torch.empty(0, 0))
            self.description_adapter = None
            self.prototype_refiner = None
            self.image_projection = None
            self.dynamic_fusion_gate = None
            self.logit_scale = None

    def _forward_backbone(self, x):
        x = self.input_enhancer(x)

        # Stage 1
        x1 = self.en_1(x)
        x1_csif = self.encoder_eca_1(self.csif_1(x1))

        # Stage 2
        x2 = self.en_2(self.pool(x1_csif))
        x2_csif = self.encoder_eca_2(self.csif_2(x2))

        # Stage 3
        x3 = self.en_3(self.pool(x2_csif))
        x3_csif = self.encoder_eca_3(self.csif_3(x3))

        # Stage 4
        x4 = self.en_4(self.pool(x3_csif))
        x4_csif = self.encoder_eca_4(self.csif_4(x4))

        # Stage 5
        x5 = self.en_5(self.pool(x4_csif))
        x5_csif = self.encoder_eca_5(self.csif_5(x5))

        # Stage 6
        x6 = self.en_6(self.pool(x5_csif))
        x6_csif = self.encoder_eca_6(self.csif_6(x6))

        # Top-down Fusion Neck
        f6 = self.decoder_eca_6(self.proj_6(x6_csif))

        p5 = self.proj_5(x5_csif)
        f5 = self.decoder_eca_5(self.ssie_5(f6, p5))

        p4 = self.proj_4(x4_csif)
        f4 = self.decoder_eca_4(self.ssie_4(f5, p4))

        p3 = self.proj_3(x3_csif)
        f3 = self.decoder_eca_3(self.ssie_3(f4, p3))

        # Multi-scale Global Pooling
        # Avg
        ap3 = F.adaptive_avg_pool2d(f3, 1).view(f3.size(0), -1)
        ap4 = F.adaptive_avg_pool2d(f4, 1).view(f4.size(0), -1)
        ap5 = F.adaptive_avg_pool2d(f5, 1).view(f5.size(0), -1)
        ap6 = F.adaptive_avg_pool2d(f6, 1).view(f6.size(0), -1)
        
        # Max
        mp3 = F.adaptive_max_pool2d(f3, 1).view(f3.size(0), -1)
        mp4 = F.adaptive_max_pool2d(f4, 1).view(f4.size(0), -1)
        mp5 = F.adaptive_max_pool2d(f5, 1).view(f5.size(0), -1)
        mp6 = F.adaptive_max_pool2d(f6, 1).view(f6.size(0), -1)

        gem3 = self.gem_pool(f3).view(f3.size(0), -1)
        gem4 = self.gem_pool(f4).view(f4.size(0), -1)
        gem5 = self.gem_pool(f5).view(f5.size(0), -1)
        gem6 = self.gem_pool(f6).view(f6.size(0), -1)

        pooled = torch.cat([ap3, mp3, ap4, mp4, ap5, mp5, ap6, mp6], dim=1)
        scale_tokens = torch.stack(
            [ap3, mp3, ap4, mp4, ap5, mp5, ap6, mp6, gem3, gem4, gem5, gem6],
            dim=1,
        )
        scale_summary, refined_tokens, scale_token_weights, scale_attention_map = self.scale_attention(scale_tokens)
        gem_features = torch.stack([gem3, gem4, gem5, gem6], dim=1)
        enhanced_pooled = torch.cat([pooled, scale_summary, gem3, gem4, gem5, gem6], dim=1)
        feature_maps = {
            "encoder": [self.en_1, self.en_2, self.en_3, self.en_4, self.en_5, self.en_6],
            "decoder": [self.proj_6, self.ssie_5, self.ssie_4, self.ssie_3],
            "fused": {"f3": f3, "f4": f4, "f5": f5, "f6": f6},
        }
        backbone_aux = {
            "scale_summary": scale_summary,
            "scale_tokens": refined_tokens,
            "scale_token_weights": scale_token_weights,
            "scale_attention_map": scale_attention_map,
            "gem_features": gem_features,
        }
        return pooled, enhanced_pooled, feature_maps, backbone_aux

    def _description_logits(self, enhanced_pooled):
        image_embed = self.image_projection(enhanced_pooled)
        image_embed = F.normalize(image_embed, dim=1)

        adapted_description = self.description_adapter(self.description_embeddings)
        adapted_description = adapted_description + self.description_embeddings
        adapted_description, prototype_relation = self.prototype_refiner(adapted_description)
        adapted_description = F.normalize(adapted_description, dim=1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * image_embed @ adapted_description.t()
        return logits, image_embed, adapted_description, prototype_relation

    def forward(self, x, return_aux=False):
        pooled, enhanced_pooled, feature_maps, backbone_aux = self._forward_backbone(x)
        base_visual_logits = self.classifier(pooled)
        enhanced_visual_logits = self.enhanced_classifier(enhanced_pooled)
        visual_logits = base_visual_logits + (self.visual_residual_scale * enhanced_visual_logits)

        if not self.use_description_aux:
            if return_aux:
                return visual_logits, {
                    "visual_logits": visual_logits,
                    "base_visual_logits": base_visual_logits,
                    "enhanced_visual_logits": enhanced_visual_logits,
                    "pooled_features": pooled,
                    "enhanced_pooled_features": enhanced_pooled,
                    "feature_maps": feature_maps,
                    **backbone_aux,
                }
            return visual_logits

        description_logits, image_embed, adapted_description, prototype_relation = self._description_logits(enhanced_pooled)
        fusion_gate = self.dynamic_fusion_gate(enhanced_pooled, image_embed, description_logits)
        mix_gate = self.description_mix_weight * (0.5 + fusion_gate)
        logits = visual_logits + (mix_gate * description_logits)

        if not return_aux:
            return logits

        aux = {
            "visual_logits": visual_logits,
            "base_visual_logits": base_visual_logits,
            "enhanced_visual_logits": enhanced_visual_logits,
            "description_logits": description_logits,
            "image_embedding": image_embed,
            "description_embedding_bank": adapted_description,
            "pooled_features": pooled,
            "enhanced_pooled_features": enhanced_pooled,
            "fusion_gate": fusion_gate,
            "mix_gate": mix_gate,
            "feature_maps": feature_maps,
            "prototype_relation_matrix": prototype_relation,
            **backbone_aux,
        }
        return logits, aux

if __name__ == '__main__':
    # Unit test shape propagation
    model = AGSENetClassifier()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)
    assert y.shape == (2, 5), "Output shape should be [2, 5]"
