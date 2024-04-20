import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                alpha=torch.tensor (alpha)
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        #print("0",ids.data.view(-1))
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from evaluation.GlobalCenterLoss import GlobalCenterTriplet
from evaluation.loss import ContrastiveLoss, NTXentLoss


# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)
#
#
# # classes
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super(PreNorm,self).__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.):
#         super(FeedForward,self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super(Attention,self).__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#
#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
#
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
#         super(Transformer,self).__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
#             ]))
#
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x


# class transformer_select(nn.Module):
#     def __init__(self,number_class ,image_size=224, patch_size=16, dim=768, depth=6, heads=8, mlp_dim=768*2,
#                  pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
#         super(transformer_select, self).__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)
#
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#
#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
#             nn.Linear(patch_dim, dim),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#         self.last_layer = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout)
#
#         self.pool = pool
#         self.to_latent = nn.Identity()
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, number_class)
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 #nn.init.constant_(m.bias, 0)
#
#
#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
#         noise = torch.rand(N, L, device=x.device)
#         # noise in [0, 1]
#         # print("noise",noise)
#         # sort noise for each sample #返回排序后的值所对应原a的下标，即torch.sort()返回的indices
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove#从小到大顺序
#         # print("ids_shuffle",ids_shuffle)
#         # print("ids_shuffle", ids_shuffle.shape)
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#         # print("ids_restore",ids_restore)
#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # 按索引提出 索引几行几列列数是值
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#         return x_masked, mask, ids_restore
#
#     def forward_feature(self, img, mask_ratio):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape
#         cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
#         # print(cls_tokens)
#         x = torch.cat((cls_tokens, x), dim=1)
#         # print("0",x.shape)
#         # print("1",self.pos_embedding[:, :(x.shape[1])])
#         x += self.pos_embedding[:, :(x.shape[1])]
#         # x = self.dropout(x)
#         x = self.transformer(x)
#         x_mask = x[:, 1:, :]
#         token = x[:, 0]
#         if mask_ratio != 0:
#             x0, mask0, ids_restore0 = self.random_masking(x_mask, mask_ratio)
#             x = torch.cat((x[:, :1, :], x0), dim=1)
#         x = self.last_layer(x)
#         token0 = x[:, 0]
#         x0 = token0#+tokent
#         return x0,token0
#
#     def forward_classifier(self, x):
#         x = self.to_latent(x)
#         x=self.mlp_head(x)
#         # x=F.softmax(x,dim=-1)
#         return x
# class transformer_select(nn.Module):
#     def __init__(self):
#         super(transformer_select, self).__init__()
#
#         self.vit = ViT()
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#
#     def forward(self, x, ratio):
#         # x=self.backbone(x)
#         # print(x.shape)
#         x = self.vit.forward_feature(x, ratio)
# #         return x
# class T_Effective_coding(nn.Module):
#     def __init__(self, num_classes: object) -> object:
#         super(T_Effective_coding, self).__init__()
#
#         self.T_O_motion = transformer_select(number_class=num_classes)
#
#     def forward(self, optical_flow, ratio):
#
#         x_TO_motion,x_TO_inter= self.T_O_motion.forward_feature(optical_flow, ratio)
#         TO_SCORE = self.T_O_motion.forward_classifier(x_TO_motion)
#
#         return TO_SCORE,x_TO_inter
# #patch_size=16, embed_dim=768, depth=3, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=16,
#         mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6)
# class MaskedAutoencoderViT(nn.Module):
#     """ Masked Autoencoder with VisionTransformer backbone
#     """
#
#     def __init__(self, img_size=224, patch_size=16, in_chans=3,
#                  embed_dim=1024, depth=24, num_heads=16,
#                  decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#                  mlp_ratio=4., number_class=5, norm_layer=nn.LayerNorm, norm_pix_loss=False):
#         super().__init__()
#
#         # --------------------------------------------------------------------------
#         # MAE encoder specifics
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         num_patches = self.patch_embed.num_patches
#         self.class_num = number_class
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
#                                       requires_grad=False)  # fixed sin-cos embedding
#
#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
#             for i in range(depth)])
#         self.last_blocks = Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
#         self.norm = norm_layer(embed_dim)
#         # --------------------------------------------------------------------------
#         # --------------------------------------------------------------------------
#         # MAE decoder specifics
#         self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
#
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#
#         self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
#                                               requires_grad=False)  # fixed sin-cos embedding
#
#         self.decoder_blocks = nn.ModuleList([
#             Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
#             for i in range(decoder_depth)])
#
#         self.decoder_norm = norm_layer(decoder_embed_dim)
#         self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
#         # --------------------------------------------------------------------------
#
#         self.norm_pix_loss = norm_pix_loss
#         # self.mlp_head = nn.Sequential(
#         #     nn.LayerNorm(embed_dim),
#         #     nn.Linear(embed_dim, self.class_num)
#         # )
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
#                                             cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#         decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
#                                                     int(self.patch_embed.num_patches ** .5), cls_token=True)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
#
#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=.02)
#         torch.nn.init.normal_(self.mask_token, std=.02)
#
#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def patchify(self, imgs):
#         """
#         imgs: (N, 3, H, W)
#         x: (N, L, patch_size**2 *3)
#         """
#         p = self.patch_embed.patch_size[0]
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#
#         h = w = imgs.shape[2] // p
#         x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
#         x = torch.einsum('nchpwq->nhwpqc', x)
#         x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
#         return x
#
#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size**2 *3)
#         imgs: (N, 3, H, W)
#         """
#         p = self.patch_embed.patch_size[0]
#         h = w = int(x.shape[1] ** .5)
#         assert h * w == x.shape[1]
#
#         x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
#         return imgs
#
#     def random_masking(self, x, mask_ratio):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
#
#         noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
#
#         # sort noise for each sample
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#
#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#
#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#
#         return x_masked, mask, ids_restore
#
#     def forward_encoder(self, x):
#
#         x = self.patch_embed(x)
#
#         # add pos embed w/o cls token
#         x = x + self.pos_embed[:, 1:, :]
#         # masking: length -> length * mask_ratio
#         # append cls token
#         cls_token = self.cls_token + self.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         # apply Transformer blocks
#         for blk in self.blocks:
#             x = blk(x)
#         # x = self.norm(x)
#
#         return x
#
#     def forward_decoder(self, x, ids_restore):
#
#         x = x + self.decoder_pos_embed
#
#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)
#
#         # predictor projection
#         x = self.decoder_pred(x)
#
#         # remove cls token
#         x = x[:, 1:, :]
#
#         return x
#
#     def forward_loss(self, imgs, pred, mask):
#         """
#         imgs: [N, 3, H, W]
#         pred: [N, L, p*p*3]
#         mask: [N, L], 0 is keep, 1 is remove,
#         """
#         target = self.patchify(imgs)
#         if self.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6) ** .5
#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#         return loss
#
#     def forward_constract_loss(self, pred0, pred1, temperature=0.1):
#         normalized_rep1 = F.normalize(pred0)
#         normalized_rep2 = F.normalize(pred1)
#         dis_matrix = torch.mm(normalized_rep1, normalized_rep2.T) / temperature
#
#         pos = torch.diag(dis_matrix)
#         dedominator = torch.sum(torch.exp(dis_matrix), dim=1)
#         loss = (torch.log(dedominator) - pos).mean()
#         return loss
#
#     def forward(self, imgs, mask_ratio=0.75):
#         global x, loss
#         pred_m = []
#         x_cls_m0 = []
#         loss_m = []
#         x = self.forward_encoder(imgs)
#
#         x_mask = x[:, 1:, :]
#         x, mask, ids_restore = self.random_masking(x_mask, mask_ratio)
#         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
#
#         x_ = torch.cat([x, mask_tokens], dim=1)
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
#         x = torch.cat([x[:, :1, :], x_], dim=1)
#         #
#         decoder_x = self.decoder_embed(x)
#
#         # decoder_x = self.decoder_norm(decoder_x)
#         pred = self.forward_decoder(decoder_x, ids_restore)
#         loss = self.forward_loss(imgs, pred, mask)
#         return loss, pred, mask, imgs
#
#     def forward_class_train(self, imgs, mask_ratio=0.75):
#         global x, loss, pred_cl
#         pred_m = []
#         x_cls_m0 = []
#         loss_m = []
#         x = self.forward_encoder(imgs)
#
#         depth = 2
#         for i in range(depth):
#             x_mask = x[:, 1:, :]
#             x_class = x[:, :1, :]
#             x, mask, ids_restore = self.random_masking(x_mask, mask_ratio)
#             #     # append mask tokens to sequence
#             mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
#
#             x_ = torch.cat([x, mask_tokens], dim=1)
#             x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
#             x = torch.cat([x_class, x_], dim=1)
#
#             decoder_x = self.decoder_embed(x)
#
#             decoder_x = self.decoder_norm(decoder_x)
#             m = nn.AdaptiveAvgPool2d((1, 1))
#             x_cls_m = x[:, 0]
#
#             pred = self.forward_decoder(decoder_x, ids_restore)  # [N, L, p*p*3]
#
#             loss = self.forward_loss(imgs, pred, mask)
#             pred = m(pred)
#             pred = pred.view(pred.shape[1], -1)
#             pred = pred.unsqueeze(0)
#             pred_m.append(pred)
#             x_cls_m = x_cls_m.unsqueeze(0)
#             x_cls_m0.append(x_cls_m)
#             loss = loss.unsqueeze(0)
#             loss_m.append(loss)
#
#         pred_m= torch.cat(pred_m, dim=0)
#
#         con_loss_f=NTXentLoss
#         loss_f= con_loss_f(pred_m[0],pred_m[1])
#         x_cls_m0 = torch.cat( x_cls_m0, dim=0)
#         loss_m = torch.cat(loss_m, dim=0)
#         x = self.last_blocks(x)
#         x_cls = x[:, 0]
#
#         # loss_f=self.forward_con_loss(pred_m[0],pred_m[1])
#
#         loss = loss_m[0] + loss_m[1]
#         return loss, x_cls  # ,x_cls_m        0
#
#     def forward_class_test(self, imgs, mask_ratio):
#         x = self.forward_encoder(imgs)
#         x = self.last_blocks(x)
#         x_cls = x[:, 0]
#
#         return x_cls
#
class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4.,number_class=5,norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.class_num=number_class
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.last_blocks=Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.class_num)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):

        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x=self.norm(x)

        return x



    def forward_decoder(self, x, ids_restore):

        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    def forward_con_loss(self,pred0,pred1):
        loss = (pred0 - pred1) ** 2
        loss = loss.mean(dim=-1)
        return loss
    def forward_constract_loss(self,pred0,pred1,temperature=0.1):
        normalized_rep1 = F.normalize(pred0)
        normalized_rep2 = F.normalize(pred1)
        dis_matrix = torch.mm(normalized_rep1, normalized_rep2.T) / temperature

        pos = torch.diag(dis_matrix)
        dedominator = torch.sum(torch.exp(dis_matrix), dim=1)
        loss = (torch.log(dedominator) - pos).mean()
        return loss
    def forward(self, imgs, mask_ratio=0.75):
        global x, loss
        pred_m=[]
        x_cls_m0=[]
        loss_m=[]
        x = self.forward_encoder(imgs)
        # if mask_ratio!=0:
        #     depth=2
        #     for i in range(depth):
        #         x_mask = x[:, 1:, :]
        #         x, mask, ids_restore = self.random_masking(x_mask, mask_ratio)
        #     #     # append mask tokens to sequence
        #         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        #
        #         x_ = torch.cat([x, mask_tokens], dim=1)
        #
        #         x = torch.cat([x[:, :1, :], x_], dim=1)
        #         #
        #         decoder_x=self.decoder_embed(x)
        #
        #         decoder_x = self.decoder_norm(decoder_x)
        #         m = nn.AdaptiveAvgPool2d((1, 1))
        #         x_cls_m = x[:, 0]
        #
        #         pred = self.forward_decoder(decoder_x, ids_restore)  # [N, L, p*p*3]
        #
        #         loss = self.forward_loss(imgs, pred, mask)
        #         pred = m(pred)
        #         pred = pred.view(pred.shape[1], -1)
        #         pred = pred.unsqueeze(0)
        #         pred_m.append(pred)
        #         x_cls_m = x_cls_m.unsqueeze(0)
        #         x_cls_m0.append(x_cls_m)
        #         loss=loss.unsqueeze(0)
        #         loss_m.append(loss)


            # pred_m= torch.cat(pred_m, dim=0)

            # con_loss_f=NTXentLoss()
            # loss_f= con_loss_f(pred_m[0],pred_m[1])
            # x_cls_m0 = torch.cat( x_cls_m0, dim=0)
            # loss_m = torch.cat(loss_m, dim=0)
            # x = self.last_blocks(x)
            # x_cls= x[:, 0]
            # pred_cl= self.mlp_head(x_cls)
            # loss_f=self.forward_con_loss(pred_m[0],pred_m[1])
            #
            # loss=loss_m[0]+loss_m[1]
        x_mask = x[:, 1:, :]
        x_cls=x[:, :1, :]
        #x[:, :1, :]
        x, mask, ids_restore = self.random_masking(x_mask, mask_ratio)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x_cls, x_], dim=1)
       #
        decoder_x=self.decoder_embed(x)

        # decoder_x = self.decoder_norm(decoder_x)
        pred = self.forward_decoder(decoder_x, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss,pred,mask,imgs
    # def forward_img(self,img,pred,mask):
    #     # print("###1",img.shape)
    #     # print("###2",pred.shape)
    #     # print("###3", x_.shape)
    #    # img=self.unpatchify(img)
    #     pred=self.unpatchify(pred)
    #     x_ = self.unpatchify(x_)
    #     return img,pred,x_

    def forward_class(self, imgs, mask_ratio=0.75):
        global x, loss, pred_cl
        pred_m=[]
        x_cls_m0=[]
        loss_m=[]
        # x = self.patch_embed(imgs)
        #
        # # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]
        # # masking: length -> length * mask_ratio
        # # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)
        x = self.forward_encoder(imgs)
        if mask_ratio!=0:
            depth=2
            for i in range(depth):
                x_mask = x[:, 1:, :]
                x_cls = x[:, :1, :]
                x, mask, ids_restore = self.random_masking(x_mask, mask_ratio)
                #print("####",)
            #     # append mask tokens to sequence
                mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

                x_ = torch.cat([x, mask_tokens], dim=1)
                x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
                x = torch.cat([x_cls, x_], dim=1)
                #x=torch.cat([x_cls,x_],dim=1)
                #
                decoder_x=self.decoder_embed(x)

                decoder_x = self.decoder_norm(decoder_x)
                m = nn.AdaptiveAvgPool2d((1, 1))
                x_cls_m = x[:, 0]

                pred = self.forward_decoder(decoder_x, ids_restore)  # [N, L, p*p*3]

                loss = self.forward_loss(imgs, pred, mask)
                pred = m(pred)
                pred = pred.view(pred.shape[1], -1)
                pred = pred.unsqueeze(0)
                pred_m.append(pred)
                x_cls_m = x_cls_m.unsqueeze(0)
                x_cls_m0.append(x_cls_m)
                loss=loss.unsqueeze(0)
                loss_m.append(loss)

            x_cls_m0 = torch.cat( x_cls_m0, dim=0)
            loss_m = torch.cat(loss_m, dim=0)
            x = self.last_blocks(x)
            x_cls= x[:, 0]
            pred_cl= self.mlp_head(x_cls)

            loss=loss_m[0]+loss_m[1]
            return loss, pred_cl,x_cls
        else:
            x = self.last_blocks(x)
            x_cls = x[:, 0]
            pred_cl = self.mlp_head(x_cls)
            return x_cls

class Model(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, number_class=5,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(Model, self).__init__()
        self.backbone = MaskedAutoencoderViT(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
                                             decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio,
                                             number_class, norm_layer, norm_pix_loss)
        self.memory = None
        self.out_dim = embed_dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, number_class)
        )
        self.id_loss = nn.CrossEntropyLoss()
        self.training = None
        self.global_center = None
        self.gc_loss = GlobalCenterTriplet(0.3, channel=3)
        self.classification = None
        self.mem = None
        self.bn_neck = nn.BatchNorm1d(self.out_dim)
        nn.init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

#     def forward(self, inputs):
#         print("$$$$",inputs.shape)
#         loss, pred_cl,x_cls_m0
#         if mask_ratio != 0:
#             loss_d, feats = self.backbone.forward_class_train(inputs, mask_ratio)
#             # print("EEE",feats.shape)
#             if not self.training:
#                 # print('m')
#
#                 return F.normalize(feats, dim=1)
#             else:
#                 # print('train')
#
#                 return self.train_forward(feats, labels, loss_d)
#         else:
#             # print('test')
#             loss = 0
#             #feats = self.backbone.forward_class_test(inputs, mask_ratio)
#
#             x = self.backbone.forward_encoder(inputs)
#             x = self.backbone.last_blocks(x)
#             feats= x[:, 0]
#             logits = self.mlp_head(feats)
#             # cls_loss = self.id_loss(logits.float(), labels)
#             # loss += cls_loss
#             return feats, logits#, loss
#             #return #self.test_forward(feats, labels)           #为了可视化输出特征
# #为了最终的可视化
#     # def train_forward(self, feats, labels, loss_d):
#     #     global logits
#     #
#     #     loss = 0
#     #     if self.classification:
#     #         logits = self.mlp_head(feats)
#     #         cls_loss = self.id_loss(logits.float(), labels)
#     #         loss += cls_loss+0.5*loss_d
#     #         print("c", cls_loss)
#     #         print("id", loss_d)
#     #     if self.mem:
#     #         feat = F.normalize(feats, dim=1)
#     #         cls_loss,logits = self.memory(feat, labels, return_logits=True)
#     #         loss += 0.5*cls_loss
#     #         print("m",cls_loss)
#     #     print("total",loss)
#     #     return logits, loss
#     #
#     # def test_forward(self, feats, labels):
#     #     global logits
#     #     loss = 0
#     #
#         # logits = self.mlp_head(feats)
#         # cls_loss = self.id_loss(logits.float(), labels)
#         # loss += cls_loss
#         # return feats,logits, loss
    def forward(self, inputs, mask_ratio, labels):
        # print("$$$$",inputs.shape)
        # loss, pred_cl,x_cls_m0
        if mask_ratio != 0:
            loss_d, feats = self.backbone.forward_class_train(inputs, mask_ratio)
            # print("EEE",feats.shape)
            if not self.training:
                # print('m'
                return F.normalize(feats, dim=1)
            else:
                # print('train')
                 return self.train_forward(feats, labels, loss_d)
        else:
            print('test')
            feats = self.backbone.forward_class(inputs, mask_ratio)
            return self.test_forward(feats, labels)

    # def train_forward(self, feats, labels, loss_d):
    #     global logits
    #
    #     loss = 0
    #     if self.classification:
    #         logits = self.mlp_head(feats)
    #         cls_loss = self.id_loss(logits.float(), labels)
    #         loss += cls_loss+0.5*loss_d
    #         # print("c", cls_loss)
    #         # print("id", loss_d)
    #     if self.mem:
    #         feat=F.normalize(feats,dim=1)
    #         cls_loss,logits=self.memory(feat,labels,return_logits=True)
    #         loss+=0.5*cls_loss
    #         print("m",0.5*cls_loss)
    #     print("total",loss)
    #     return logits, loss
    def train_forward(self, feats, labels, loss_d):
        global logits, loss_MCL, loss_cls

        loss = 0
        loss_d_total=0
        loss_MCL_total=0
        loss_cls_total=0
        if self.classification:
            logits = self.mlp_head(feats)
            loss_cls = self.id_loss(logits.float(), labels)
            loss += loss_cls+0.5*loss_d
            loss_d_total+=0.5*loss_d
            loss_cls_total+=loss_cls
            # print("c",  loss_cls)
            # print("id", loss_d)
        if self.mem:
            feat = F.normalize(feats, dim=1)
            loss_MCL,logits0 = self.memory(feat, labels, return_logits=True)
            loss += 0.5 *loss_MCL
            loss_MCL_total+=0.5 *loss_MCL
            # print("m",loss_MCL)
        # print("total",loss)
        return logits, loss#, loss_d_total,loss_cls_total,loss_MCL_total,feats

    def test_forward(self, feats, labels):
        global logits
        loss = 0

        logits = self.mlp_head(feats)
        cls_loss = self.id_loss(logits.float(), labels)
        loss += cls_loss
        return logits, loss


