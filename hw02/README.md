# Engineering-practices-ml

# Установка пакетного менеджера
# Развертывание окружения
Для разработки:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements_dev.txt
```

Для использования:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
# Сборка пакета
```
python3 -m build
twine upload -r testpypi dist/*
```
# Ссылка на пакет в pypi-test
https://test.pypi.org/project/hw-package-ml/1.0/
# Установка пакета из pypi-test
```
pip install -i https://test.pypi.org/simple/ hw-package-ml==1.2
```