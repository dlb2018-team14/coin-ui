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

[注意] .backupという拡張子だと途中でエラーになる（darkflowがいけてない）。.weightsという拡張子に変更しないとだめ

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

## 実行（yolo v3)

```
python3 ./v3_main.py
```

## 実行(yolo v2)

```
python3 ./v2_main.py

# AssertionError: expect 268365952 bytes, found 268365956
# のように4バイトだけ違うエラーが出たら、darkflow/darkflow/utils/loader.pyを書き換える
# https://qiita.com/wikipediia/items/0dc4574dc137e11bb6be
# https://github.com/thtrieu/darkflow/issues/421
sed -i '.bak' 's/self.offset = 16/self.offset = 20/g' ./darkflow/darkflow/utils/loader.py

# 画像出力
# data/sample-images/out/配下にバウンディングボックスつきの画像が出力される
python3 ./v2_main_save_image.py

# webcam経由でリアルタイム出力
python3 ./v2_main_camera.py
```

