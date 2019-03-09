# coin-ui

dl基礎講座 team14 プロジェクトのユーザーインターフェス

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
