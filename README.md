# Auto-completion of the text with LSTM
LSTM предсказывает следующий токен, обучаясь на датасете из твитов на английском языке

## Структура проекта

text-autocomplete/
├── data/                            # Датасеты
│   ├── raw_dataset.csv              # "сырой" скачанный датасет
│   └── dataset_processed.csv        # "очищенный" датасет
│   └── tokenized_dataset.csv        # токенизированный датасет для LSTM
│   ├── train.csv                    # тренировочная выборка
│   ├── val.csv                      # валидационная выборка
│   └── test.csv                     # тестовая выборка
│
├── src/                             # Весь код проекта
│   ├── data_utils.py                # Обработка датасета
|   ├── next_token_dataset.py        # код с torch Dataset'ом 
│   ├── lstm_model.py                # код lstm модели
|   ├── eval_lstm.py                 # замер метрик lstm модели
|   ├── lstm_train.py                # код обучения модели
|   ├── eval_transformer.py          # код с запуском и замером качества трансформера
│
├── configs/                         # yaml-конфиг в котором происходит настройка параметров обучения, модели и путей к данным/сохранениям
│
├── models/                          
│   ├── lstm_next_token.pt           # веса обученных моделей
|   ├── loss_curves.png              # график с train_loss, val_loss значениями во время обучения 
|
├── solution.ipynb                   # ноутбук с решением и описанием
└── requirements.txt                 # зависимости проекта 