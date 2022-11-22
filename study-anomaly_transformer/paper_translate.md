## 論文メモ

### 1. Introduction
* 異常値は稀でラベリング自体も高コストなので、教師なしの手法を確立したい(Unsupervised time series anomaly detection)
* 先行研究
    * 確率密度の推定？
    * クラスタリングベースの手法(SVMとか)
    ⇒ これらは時系列要素がないのと、未知の異常値に対応できない
    * ニューラルネット系は非常にWellではある。
        * pointwise representationに着目
        * RNUとかから自己回帰を求めるイメージ
        * そういう手法はたいていAnomaly Criterionにpointwise reconstructionかprediction errorを用いている。
        * ただ、そもそも論として異常点がデータに少ない場合にpointwise reconstructionでは厳しい場合アリ
        * あと、点別でやっていくとtemporal contextが落ちます
            * !! RNUとかLSTMはそれを加味できます、って話じゃなかったっけ？
                * 遠いやつほど情報が落ちる、ってことだった気もする。Transformerならたしかにwindowに入れれば全部加味できる。
        * もう一つの手法は状態空間モデル系。
            * GNN(Graph Neural Network)はアツい。
            * でもやっぱりsingle time pointにstricted
* なので、Transformerを使います。
* "Applying Transformers to time series, we find that <the temporal association of each time point\> can be obtained
    * from the self-attention map,
        * which presents as a distribution of its association weights to all the time points along the temporal dimension.
* なんかよく分からんけど、temporal associationが要するに時間方向のcontext的な意味っぽい。
* "Series-association"と名付けられた。何が？
    * まあなんか関係性みたいな話な気はする。raw series of transformersと関連アリ
* 異常値が少ないために、何も考えずに計算すると隣接点の情報に下駄を履かせてしまう（adjacent-concentration bias)
* このバイアスをprior-associationと呼ぶ。
* 一方、通常点は全体系との関連付けをうまく計算できる
* つまり、これって潜在的には異常点と通常点を区別できているということにならない？（inherent distinguishablity)
* これを異常度の基準としたらめっちゃいい感じだと思う。つまり、各時刻点の距離をseries-associationとprior-associationで定義したAssociation Discripancyを定義する。
    * AssDisが小さいほど異常
    * なぜなら隣接点バイアスがあるため

### 3. Method
