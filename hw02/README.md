# Engineering-practices-ml

## Пакетный менеджер
Рекомендуется использовать pip
## Oкружение
dev:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements_dev.txt
```

prod:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
## Сборка пакета
```
python3 -m build
twine upload -r testpypi dist/*
```
## Ссылка на пакет в pypi-test
https://test.pypi.org/project/hw-package-ml/1.2/
## Установка пакета из pypi-test
```
pip install -i https://test.pypi.org/simple/ hw-package-ml==1.2
```