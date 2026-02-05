import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViTForJaguarReID(nn.Module):
    def __init__(self, num_classes=31, image_size=384, patch_size=32, dim=1024,
                 depth=12, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        # Patch embedding
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        )

        # Position embedding and class token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout)

        # Re-ID specific heads
        self.reid_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Linear(512, num_classes)

        # ArcFace layer for metric learning
        self.arcface = ArcFace(512, num_classes, s=30.0, m=0.5)

        self.pool = 'cls'

    def forward_features(self, x):
        # Patch embedding
        x = self.to_patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        b, n, _ = x.shape

        # Add class token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Pooling
        if self.pool == 'cls':
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return x

    def forward(self, x, labels=None, return_features=False):
        # Extract features
        features = self.forward_features(x)

        # Re-ID features
        reid_features = self.reid_head(features)

        if return_features:
            return reid_features

        # Classification
        if labels is not None:
            # Use ArcFace for training
            logits = self.arcface(reid_features, labels)
        else:
            # Use standard classifier for inference
            logits = self.classifier(reid_features)

        return logits, reid_features


class ArcFace(nn.Module):
    """ArcFace loss layer for metric learning"""

    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Normalize features and weights
        input_norm = F.normalize(input)
        weight_norm = F.normalize(self.weight)

        # Cosine similarity
        cosine = F.linear(input_norm, weight_norm)

        # Add margin
        phi = cosine - self.m

        # One-hot encoding
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # Combine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# Alternative: Using pre-trained models
class MegaDescriptorWrapper(nn.Module):
    """Wrapper for MegaDescriptor foundation model"""

    def __init__(self, num_classes=31, model_size='base'):
        super().__init__()

        # Load pre-trained MegaDescriptor
        if model_size == 'small':
            self.backbone = timm.create_model('tf_efficientnet_b0', pretrained=True)
            feature_dim = 1280
        elif model_size == 'base':
            self.backbone = timm.create_model('tf_efficientnet_b2', pretrained=True)
            feature_dim = 1408
        else:  # large
            self.backbone = timm.create_model('tf_efficientnet_b4', pretrained=True)
            feature_dim = 1792

        # Re-ID head
        self.reid_head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone.forward_features(x)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)

        reid_features = self.reid_head(features)

        if return_features:
            return reid_features

        logits = self.classifier(reid_features)
        return logits, reid_features