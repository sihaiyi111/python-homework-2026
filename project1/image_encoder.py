"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """

        # TODO-02 contract: [B,N,D] to [B,H,N,D/H].
        # Image model: D=768, H=12, D/H=64.
        B,N,D=x.shape
        H=self.heads
        return x.reshape(B,N,H,D//H).transpose(0,2,1,3)
        # 将每个 token 的 D 个特征拆成 H 组，每组 D//H ，得到 [B, N, H, D//H]。
        # 交换 H 和 N 两个轴，得到 [B, H, N, D//H]，按每个头计算注意力。
        #Attention->每个头分别计算 Q [B,H,N,64] @ K转置 [B,H,64,N] → 分数 [B,H,N,N]


    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        scores=q@k.transpose(0,1,3,2)/np.sqrt(np.float32(self.head_dim))

        # Q与K的转置相乘,得到[B,H,N,N]
        # 除以每头特征维度的平方根，避免分数差距过大，让 softmax 的权重过于集中。（ai告诉我的）



        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        
        

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        context=coefficients@v
        B,H,N,head_dim=context.shape
        merged=context.transpose(0,2,1,3).reshape(B,N,H*head_dim)

        # 用coefficient@v，再把同一token的各头特征拼起来
        # [B,H,N,64]->[B,N,H,64]->[B,N,D]

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        return merged@self.out_weight.T+self.out_bias
    
        # Attention->拼起来的各头特征@输出权重.T+bias
        # 权重存[输出,输入],所以右乘时要转置，结果还是[B,N,D]

class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        residual=update+x
        # Attention->update,把它加到原来的x上,得到residual残差


        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].

        hidden=self.gelu(u@self.fc1_weight.T+self.fc1_bias)
        mlp_update=hidden@self.fc2_weight.T+self.fc2_bias

        return residual+mlp_update
    
        # mlp= GELU(u @ W1.T + b1) @ W2.T + b2.用hidden储存临时结果
        #return 残差+mlpupdate

class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # Input [B,3,224,224], return [B,49,3072].
        B,C,H,W=images.shape
        return images.reshape(B,C,7,32,7,32).transpose(0,2,4,1,3,5).reshape(B,49,3072)
        #原来224*224 拆成7*7小块，每个小块像素32*32.再把3通道合起来,就是32*32*3=3072.
        #先把224,224 reshape成7,32,7,32 再transpose [B,7,7,3,32,32],最后reshape[B,49,3072]

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        tokens=patches@self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        tokens=np.concatenate((cls,tokens),axis=1)
        return tokens+self.position
        

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        cls=tokens[:, 0, :]
        cls=self.post_norm(cls)
        output=cls@self.projection

        return unit_vector(output)
        # 先[:, 0, :]取每张图片的第0个token, CLS → [B,768]
        # 先做post_norm,再乘projection→[B,512]最后把每行归一化