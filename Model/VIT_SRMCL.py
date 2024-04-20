import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class MaskedAutoencoderViT_SR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., number_class=5, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.class_num = number_class
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.last_blocks = Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
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
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, self.class_num)
        # )

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
        x = self.norm(x)

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

    def forward_constract_loss(self, pred0, pred1, temperature=0.1):
        normalized_rep1 = F.normalize(pred0)
        normalized_rep2 = F.normalize(pred1)
        dis_matrix = torch.mm(normalized_rep1, normalized_rep2.T) / temperature

        pos = torch.diag(dis_matrix)
        dedominator = torch.sum(torch.exp(dis_matrix), dim=1)
        loss = (torch.log(dedominator) - pos).mean()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        global x, loss

        x = self.forward_encoder(imgs)

        x_mask = x[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x_mask, mask_ratio)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        #
        decoder_x = self.decoder_embed(x)

        # decoder_x = self.decoder_norm(decoder_x)
        pred = self.forward_decoder(decoder_x, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, imgs

    def forward_class_train(self, imgs, mask_ratio=0.75,depth=2):
        global x, loss, pred_cl
        pred_m = []
        x_cls_m0 = []
        loss_m = []
        x = self.forward_encoder(imgs)
        for i in range(depth):
            x_mask = x[:, 1:, :]
            x_class=x[:, :1, :]
            x, mask, ids_restore = self.random_masking(x_mask, mask_ratio)
            #     # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

            x_ = torch.cat([x, mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = torch.cat([x_class, x_], dim=1)

            decoder_x = self.decoder_embed(x)

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
            loss = loss.unsqueeze(0)
            loss_m.append(loss)


        loss_m = torch.cat(loss_m, dim=0)
        x = self.last_blocks(x)
        x_cls = x[:, 0]
        loss = loss_m[0] + loss_m[1]
        return loss, x_cls

    def forward_class_test(self, imgs, mask_ratio):
        x = self.forward_encoder(imgs)
        x = self.last_blocks(x)
        x_cls = x[:, 0]

        return x_cls


class Model(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4, number_class=5,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(Model, self).__init__()
        self.backbone = MaskedAutoencoderViT_SR(img_size, patch_size, in_chans, embed_dim, depth, num_heads,
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

        self.classification = None
        self.mem = None
        self.bn_neck = nn.BatchNorm1d(self.out_dim)
        nn.init.constant_(self.bn_neck.bias, 0)
        self.bn_neck.bias.requires_grad_(False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs, mask_ratio, labels,depth):

        if mask_ratio != 0:
            loss_d, feats = self.backbone.forward_class_train(inputs, mask_ratio,depth)
            if not self.training:
                return F.normalize(feats, dim=1)
            else:
                return self.train_forward(feats, labels, loss_d)
        else:
            feats = self.backbone.forward_class_test(inputs, mask_ratio)
            return self.test_forward(feats, labels)
    def train_forward(self, feats,labels, loss_d,a=0.5,b=0.5):
        global logits
        loss = 0
        if self.classification:
            logits = self.mlp_head(feats)
            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss+a*loss_d
        if self.mem:
            feat=F.normalize(feats,dim=1)
            cls_loss,logits=self.memory(feat,labels,return_logits=True)
            loss+=b*cls_loss
        return logits, loss
    def test_forward(self, feats, labels):
        global logits
        loss = 0
        logits = self.mlp_head(feats)
        cls_loss = self.id_loss(logits.float(), labels)
        loss += cls_loss
        return feats,logits, loss
