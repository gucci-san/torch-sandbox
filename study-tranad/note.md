# TranAD<br><br>
## TranADとは
* Transformerを使って、GAN的に異常検知をやるモデル<br><br>

## 論文解読メモ<br><br>


### 3.1 Problem Formulation
#### 時系列の生データセット
$$\mathbf T = (\mathbf x_1, \mathbf x_2, ...\mathbf x_t, ... ,\mathbf x_T)$$
* ここで、すべての $ \mathbf x_t$ は $m$ 次元の列ベクトルであり、時系列のembeddingと捉える（または単に時系列のマルチモーダルデータなど）

#### 用語の定義<br>
* Univariate settings : $m=1$ の特別なパターン<br>
* Anomaly Detection : 与えられた $\mathbf T$ を学習、未知の同モード $m$ を持つ $\hat{\mathbf T} $ の各時刻に対して、$\mathbf Y = (y_1, y_2 ... y_{\hat{T}}), y \in \{0, 1\}$ を予測する。ここで、$y_t$ は各時刻ごとに「その時刻が異常かどうか」を表す二値変数。<br>
* Anomaly Diagnosis : 上記のAnomaly Detectionの問題設定において、$ y_t \in \{0, 1\}^m $ を予測する。つまり、どのモードが異常かまでを予測する。<br><br>

### 3.2 Data Processing
* min-max normalize<br>
* time series window<br>
    * 系列の始めなど、データがwindow長に足りない場合はReplicate Padding (普通にコピーするだけ)を行う
* <font color=blue>the time slice until the current timestampを $C_t$ と定義する</font> <- これがよくわからない。 $C_t$ は結局shapeどうなるんだ？

* 直接Anomaly label ($y_t$)を予測するわけではなく、まず各時刻$t$に対してAnomaly score $s_t$ を求める。
    * その $s_t$ を求めるために、まずinput window $W_t$ を再構築した $O_t$ を求め、その差分を $s_t$ の算出に利用する。
<br><br>

### 3.3 Transformer Model
* Attention, MultiheadAttentionの定義は原著通り。
* GANの文脈で、モデルを2フェーズに分ける。
* まず第一フェーズでは、$W, C$とFocus Score $F$ をinputとする。 $W$ と $F$ をconcatenateし、この入力行列をpositional encodingしたあと、1st encoderに送る。1st encoderでは
    $$ I_1^1 = LayerNorm(I_1 + MultiheadAttention(I_1, I_1, I_1)) $$
    $$ I_1^2 = LayerNorm(I_1^1 + FeedForward(I_1^1))$$
    の処理を行う。

    * では、ここでコードを見てみる。(torch_nn_TransformerEncoderLayer.py) --
    ```python
    class TranAD(nn.Module):
        .
        .
        .

    model = TranAD(**kwargs)
    z = model(window, elem)
    # window.shape : [window, batch, feats]
    # elem.shape   : [1, batch, feats]
    ```
    で、一番最初に通過するのはtorchのTransformerEncoderLayerになる。
    * 必須の引数はsrcのみ。つまり、<font color=blue>Transformerと言った時点でSelf-Attentionが前提になっている</font>ことがわかる？
    * key_padding_mask
    * why_not_sparsity_fast_path
    が何を言ってるのかは分からんが、多分本筋ではないと思うので一旦スルー。<br>
    メインの処理をやっているのは多分205行目から。
        * norm_firstはsa, ffとnormの順番を変えられるっぽい。
    ```python
    x = src
    # ...
    else:
      # I_1^1 = LayerNorm(I_1 + MultiheadAttention(I_1, I_1, I_1))
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        
      # I_1^2 = LayerNorm(I_1^1 + Feedforward(I_1^1))
        x = self.norm2(x + self._ff_block(x))
    return x
    ```
    で、
    ```python
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linaer2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    ```
    だから、式(4)の$MultiheadAttention()$がコードの_sa_block、$FeedForward()$が_ff_blockに対応していそうなことがわかる。<br><br>

* 次からが意味わかんない。
        " For the window encoder, we apply positino encoding to the input window $W$ to get $I_2$ . We modify the self-attention in the window encoder to mask the data at subsequent positions."
        まずこれがコードのどこに対応してるか全く分からん。

    * コード的には、transformer_decoder1に  
        * tgt: elem, $t$ での値 (つまり$\mathbf x_t$)
        * memory: encoderの戻り値、論文でいうと$I_1^2$に当たりそう
        が入力されている。
        * で、返ってくるのはx単騎、
        * xに対しての処理は
            ```python
            x = tgt
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = selg.norm3(x + self._ff_block(x))
            ```
            の3つ。
            * 1こめはxのself-attentionを求めている
                * x = self.attn(x, x, x)
            * 2こめはQ=x, K=V=memoryのmultiheadattentionを求めている
                * x = self.mha(x, mem, mem)
            * 3こめはfeedforward、
        * 一方、論文を順番通り読んできたときの数式は
        $$ I_2^1 = Mask(MultiheadAttention(I_2, I_2, I_2)) $$
        $$ I_2^2 = LayerNorm(I_2 + I_2^1) $$
        $$ I_2^3 = LayerNorm(I_2^2 + MultiheadAttention(I_1^2, I_1^2, I_2^2)) $$
        の3つ。
        * 100歩譲ってdecoder1が実は「window_encode」だとしても
            * Maskは明らかに使ってなさそう
            * $I_2^3$のとこ、$MultiheadAttention(I_1^2, I_1^2, I_2^2)$って書いてあるけど、コードではmha(x, mem, mem)になってる
        * なんか沼の気がしてきたな……
            * githubのissuesでもL2使ってない件とかMAMLどこやねん問題とか質問されてるけど未解決