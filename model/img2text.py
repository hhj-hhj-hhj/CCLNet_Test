import torch
import torch.nn as nn
from .clip import clip


class IMG2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

# def get_text_features(model, token_feature, clip_model, dtype):
#     device = 'cuda'
#     b = token_feature.size(0)
#     text_tokenize = clip.tokenize('A photo of X')
#     text_tokenize = text_tokenize.cuda(device, non_blocking=True)
#     text = text_tokenize.view(1, -1)
#     text = text.repeat(b, 1)
#     token_feature = token_feature.unsqueeze(1)
#     with torch.no_grad():
#         text_embedding = clip_model.token_embedding(text).type(dtype)
#     prompts = torch.cat((text_embedding[:, :-1, :], token_feature), dim=1)
#     text_features = model.text_encoder(prompts, text_tokenize)
#     return text_features

# def get_text_features(token_features, clip_model, dtype):
#     device = 'cuda'
#     text = clip.tokenize("A photo of")
#     text = text.cuda(device, non_blocking=True)
#     text = text.view(1, -1)
#     text = text.repeat(token_features.size(0), 1)
#     text_features = encode_text_img(text, token_features, clip_model, dtype)
#     return text_features

def get_text_features(token_features, clip_model, dtype, text):
    device = 'cuda'
    text = clip.tokenize(text)
    text = text.cuda(device, non_blocking=True)
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    text_features = encode_text_img(text, token_features, clip_model, dtype)
    return text_features

def get_loss_img2text(model, img2text, images, loss_img, loss_txt, clip_model, text):

    device = "cuda"
    with torch.no_grad():
        image_features = model(images, get_image=True)
    token_features = img2text(image_features)
    text_features = get_text_features(token_features, clip_model, clip_model.dtype, text)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logit_scale = clip_model.logit_scale.exp()
    logit_scale = logit_scale.mean()

    # 这是不使用分布式的代码
    ground_truth = torch.arange(len(image_features)).long()
    ground_truth = ground_truth.cuda(device, non_blocking=True)

    # Image loss.
    logits_per_image = logit_scale * image_features @ text_features.t()
    loss_img_val = loss_img(logits_per_image, ground_truth)
    logits_per_text = logit_scale * text_features @ image_features.t()
    loss_txt_val = loss_txt(logits_per_text, ground_truth)

    total_loss = (loss_img_val + loss_txt_val) / 2
    return total_loss

def get_loss(model, images, texts, loss_img, loss_txt):
    device = "cuda"

    pass

def encode_text_img(text, img_tokens, clip_model, dtype):
    b_size = img_tokens.size(0)
    x = clip_model.token_embedding(text).type(dtype)  # [batch_size, n_ctx, d_model]
    collect_ind = text == clip_model.vocab_size - 1
    collect_ind = collect_ind.nonzero()[:, 1]
    img_tokens = img_tokens.view(b_size, 1, -1)
    x = torch.cat([x[:, :collect_ind[0]], img_tokens, x[:, collect_ind[0]:-1]], dim=1)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.size(0)), collect_ind+1] @ clip_model.text_projection
    return x