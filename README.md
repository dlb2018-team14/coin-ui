# coin-ui

dl基礎講座 team14 プロジェクトのユーザーインターフェス（webカメラで硬貨画像を読み取り、リアルタイムに認識結果を表示する）

以下、macの環境構築

## （まだの場合）pyenvインストール

```
brew update
brew install pyenv
brew install pyenv-virtualenv
pyenv install --list
pyenv install 3.6.5
pyenv rehash
pyenv global 3.6.5
pyenv versions
```

.bashrcとかに以下を入れる

```
export PYENV_ROOT="${HOME}/.pyenv"
export PATH="${PYENV_ROOT}/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

## このプロジェクト用のpyenv環境構築

```
pyenv virtualenv 3.6.5 env_coin-ui
pyenv local env_coin-ui
pip install --upgrade pip
```

```
pip install -r requirements.txt
```

## darkflowをcloneしてビルド

```
git clone https://github.com/thtrieu/darkflow.git
cd ./darkflow
python3 setup.py build_ext --inplace

# ヘルプが出ることを確認
./flow --h

# pip インストールして自前のpythonプログラムから呼べるように
pip install -e .
```

## 予測に必要なデータを用意

以下のファイルをmain.pyの中で指定

```
1. 重みデータ data/weights/xxx

colabでトレーニングしてbackup/に吐き出されたファイルをダウンロード（重いのでリポジトリに入れてない）

2. コンフィグファイル data/cfg/xxx

トレーニングに利用したyoloのコンフィグファイル

3. ラベルファイル  data/labels/xxx

darkflowはデフォルトでカレントディレクトリ配下の./labels.txtをロードしようとする
ここでは、yoloのobj.namesがそのままlabels.txtに相当
このファイルがなければエラーになるし、darkflow配下にもともとあるlabels.txtはサンプル用なので今回は使えない
https://github.com/thtrieu/darkflow#parsing-the-annotations

4. 予測に使うサンプル画像 data/sample-images/xxx

任意の硬化画像

```

## 実行

```
cd ..
python3 ./main.py

# AssertionError: expect 268365952 bytes, found 268365956
# のように4バイトだけ違うエラーが出たら、darkflow/darkflow/utils/loader.pyを書き換える
# https://qiita.com/wikipediia/items/0dc4574dc137e11bb6be
# https://github.com/thtrieu/darkflow/issues/421

sed -i '.bak' 's/self.offset = 16/self.offset = 20/g' ./darkflow/darkflow/utils/loader.py

#以下のように出力されたらok

[{'label': '1yen', 'confidence': 0.919171, 'topleft': {'x': 146, 'y': 62}, 'bottomright': {'x': 289, 'y': 142}}, {'label': '1yen', 'confidence': 0.7943273, 'topleft': {'x': 784, 'y': 31}, 'bottomright': {'x': 815, 'y': 94}}, {'label': '1yen', 'confidence': 0.605291, 'topleft': {'x': 680, 'y': 117}, 'bottomright': {'x': 797, 'y': 155}}, {'label': '1yen', 'confidence': 0.90542024, 'topleft': {'x': 822, 'y': 86}, 'bottomright': {'x': 961, 'y': 177}}, {'label': '1yen', 'confidence': 0.8883557, 'topleft': {'x': 146, 'y': 159}, 'bottomright': {'x': 304, 'y': 258}}, {'label': '1yen', 'confidence': 0.81608486, 'topleft': {'x': 691, 'y': 134}, 'bottomright': {'x': 827, 'y': 229}}, {'label': '1yen', 'confidence': 0.86325437, 'topleft': {'x': 550, 'y': 207}, 'bottomright': {'x': 700, 'y': 300}}, {'label': '1yen', 'confidence': 0.8716824, 'topleft': {'x': 180, 'y': 265}, 'bottomright': {'x': 340, 'y': 373}}, {'label': '1yen', 'confidence': 0.8318603, 'topleft': {'x': 563, 'y': 281}, 'bottomright': {'x': 722, 'y': 397}}, {'label': '1yen', 'confidence': 0.9079153, 'topleft': {'x': 500, 'y': 401}, 'bottomright': {'x': 651, 'y': 527}}, {'label': '10yen', 'confidence': 0.53762823, 'topleft': {'x': 698, 'y': 14}, 'bottomright': {'x': 812, 'y': 40}}, {'label': '10yen', 'confidence': 0.37034807, 'topleft': {'x': 502, 'y': 29}, 'bottomright': {'x': 640, 'y': 88}}, {'label': '10yen', 'confidence': 0.79097235, 'topleft': {'x': 419, 'y': 169}, 'bottomright': {'x': 553, 'y': 285}}, {'label': '50yen', 'confidence': 0.6413382, 'topleft': {'x': 676, 'y': 35}, 'bottomright': {'x': 780, 'y': 98}}, {'label': '100yen', 'confidence': 0.8671419, 'topleft': {'x': 406, 'y': 52}, 'bottomright': {'x': 545, 'y': 133}}, {'label': '100yen', 'confidence': 0.77069515, 'topleft': {'x': 537, 'y': 92}, 'bottomright': {'x': 678, 'y': 186}}, {'label': '500yen', 'confidence': 0.8044342, 'topleft': {'x': 286, 'y': 80}, 'bottomright': {'x': 462, 'y': 189}}]
```

画像出力

```
python3 ./main_save_image.py

# data/sample-images/out/配下にバウンディングボックスつきの画像が出力される
```
