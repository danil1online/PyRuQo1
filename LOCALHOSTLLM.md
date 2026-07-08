# Цель

Запуск LLM на собственном локальном сервере

# Модели и примеры практического использования

- Генерация датасетов; модели (рассматриваются [Qwen3.6-35B-A3B-UD-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) для 24 VRAM и [Qwen3.5-4B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main) для 8 VRAM) загружаются в GPU.
- Работа в opencode; модель (рассматривается [Qwen3.6-35B-A3B-UD-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) для 8 VRAM) загружается в GPU (только "Experts") + CPU|RAM (остальная часть модели).

и др.

# Сервер LLM

## Параметры

Тестирование проводилось на трех ПК:

- ПК 1: Ubuntu 24.04 / Intel(R) Core(TM) i7-2600K CPU @ 2.90GHz / GTX 1070 8Gb / 32 Gb RAM / llama.cpp
- ПК 2: Ubuntu 22.04 / Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz / RTX 3050 8Gb / 256 Gb RAM / LM-Studio | llama.cpp
- ПК 3: Ubuntu 22.04 / Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz / GTX 1070 8Gb / 64 Gb RAM / llama.cpp

## Установка llama.cpp
```bash
cd ~
sudo apt update
sudo apt install build-essential git cmake ccache
```
### ПК на базе GTX 1070:

Ремарка:

Реальной поддержки F16 у GTX 1070 нет, но нужно ставить, чтобы llama.cpp могла ее эмулировать. Это позволяет использовать `--cache-type-v q8_0` (или `f16`) и разместить модель в памяти вместе со всем контекстом, на который модель способна.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8 # Стабильнее всего работает с GTX | Pascal
git clone https://github.com/ggml-org/llama.cpp.git # В последней версии уже появляется ошибка 
cd llama.cpp
# Включаем возможность использования видеокарты Nvidia Pascal
cmake -B build\
 -DGGML_CUDA=ON\
 -DGGML_CUDA_F16=ON\ 
 -DCMAKE_CUDA_ARCHITECTURES=61\
 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build --config Release -j$(nproc)
```

### ПК на базе RTX 3050 / 3090:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# Включаем возможность использования видеокарты Nvidia
cmake -B build\
 -DGGML_CUDA=ON\
 -DGGML_CUDA_F16=ON\
 -DCMAKE_CUDA_ARCHITECTURES=86\
 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build --config Release -j$(nproc)
```

## Оптимальные конфигурации запуска моделей на ограниченных ресурсах

### Запуск Qwen3.6 35B-A3B. Видеокарты с 8 Gb VRAM.

Вариант подходит для подключения моделей к opencode. Скорость генерации не большая и падает при росте контекста, но модель справляется с поставленными задачами за в целом приемлемое время.

Основная идея в условиях ограниченной VRAM: 

**Загрузка только Experts в GPU:** `-ngl 40 -ncmoe 40`

- Можно задать `--flash-attn off`, тогда
  - квантование кэша значений не получится использовать, следовательно: 
    - меньший размер контекста `-c 98304`
- Можно задать `--flash-attn on`, тогда llama.cpp на GTX 1070 будет его эмулировать программно, что существенно снизит скорость на этапе чтения кода, но есть и плюсы:
  - можно задавать `--cache-type-k q8_0 --cache-type-v q8_0` и, как следствие:
    - повысить кэш до `-c 196608`(занимает 6,7 Gb VRAM; можно `-c 262144`, но это займет 7,9 Gb VRAM, на расширение мультимодальностью mmproj не хватает)

Дополнительно рекомендуется задать:
- Количество ядер CPU для загрузки / выгрузки моделей, например, `-t 4`
- Максимальный размер пачки токенов, которую сервер может принять на вход и начать обрабатывать за один логический шаг `-b 4096` (можно `-b 8192`)
- Физический размер под-пачки, которая непосредственно отправляется на исполнение в вычислительные ядра (в CUDA на GPU или в потоки CPU) за один физический такт `-ub 1024`
- Возможность обработки изображений `--mmproj /home/user/Downloads/Q3.6-35B-mmproj-F16.gguf`

1. *Multimodal*

```bash
sudo nano /etc/systemd/system/llama-cpp.service
```

```bash
[Unit]
Description=Llama.cpp Server
After=network.target

[Service]
Type=simple

WorkingDirectory=/home/user/llama.cpp
ExecStart=/home/user/llama.cpp/build/bin/llama-server -m /home/user/Downloads/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --mmproj /home/user/Downloads/Q3.6-35B-mmproj-F16.gguf -ngl 40 -ncmoe 40 -c 196608 -b 4096 -ub 1024 -t 4 --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8079
Restart=always

[Install]
WantedBy=default.target
```

2. *Text*

```bash
sudo nano /etc/systemd/system/llama-cpp.service
```

```bash
[Unit]
Description=Llama.cpp Server
After=network.target

[Service]
Type=simple

WorkingDirectory=/home/user/llama.cpp
ExecStart=/home/user/llama.cpp/build/bin/llama-server -m /home/user/Downloads/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 40 -ncmoe 40 -c 262144 -b 4096 -ub 1024 -t 4 --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=default.target
```

### Запуск Qwen3.6 35B-A3B. Видеокарты с 24 Gb VRAM.

1. *Multimodal*

Вариант подходит для подключения моделей к opencode. Скорость генерации достаточная, но падает при росте контекста. При этом модель справляется с поставленными задачами за в целом приемлемое время.

Идея в условиях ограниченной VRAM загрузки только Experts в GPU `-ngl 40 -ncmoe 40` подходит для запуска в мультимодальном режиме, в таком случае можно использовать весь доступный модели контекст `-c 262144`.

```bash
sudo nano /etc/systemd/system/llama-cpp.service
```

```bash
[Unit]
Description=Llama.cpp Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/user/llama.cpp
ExecStart=/home/user/llama.cpp/build/bin/llama-server -m /home/user/Downloads/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf --mmproj /home/user/Downloads/mmproj-F16.gguf -ngl 40 -ncmoe 40 -c 262144 -b 8192 -ub 1024 -t 6 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8080 
Restart=always

[Install]
WantedBy=default.target
```

2. *Text*

Вариант подходит и для подключения моделей к opencode, и для генерации датасетов. Скорость генерации достаточная, но в opencode падает при росте контекста. При этом модель справляется с поставленными задачами за в целом приемлемое время.

Загрузка всей модели в GPU подходит для запуска в текстовом режиме, в таком случае следует использовать большую часть, но не весь доступный модели контекст `-c 196608`.


```bash
sudo nano /etc/systemd/system/llama-cpp-gpu.service
```

```bash
[Unit]
Description=Llama.cpp Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/user/llama.cpp
ExecStart=/home/user/llama.cpp/build/bin/llama-server -m /home/user/Downloads/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -c 196608 -b 8192 -ub 256 -t 4 --flash-attn on --timeout 600 --cache-type-k q8_0 --cache-type-v q8_0 --host 0.0.0.0 --port 8181 
Restart=always

[Install]
WantedBy=default.target
```


### Запуск Qwen3.5-4B. Видеокарты с 8 Gb VRAM.

Основное отличие -- возможность загрузки модели полностью на видеокарту, включая мультимодальные возможности модели через mmproj.

```bash
[Unit]
Description=Llama.cpp Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/user/llama.cpp
ExecStart=/home/user/llama.cpp/build/bin/llama-server -m /home/user/Downloads/Qwen3.5-4B-Q4_K_M.gguf --mmproj /home/user/Downloads/Q3.5-4B-mmproj-F16.gguf -ngl 99 -c 98304 --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=default.target
```

## Запуск и автозапуск llama.cpp

1. Простой запуск с работой "до перезагрузки ПК"

```bash
sudo systemctl daemon-reload
sudo systemctl start llama-cpp # или llama-cpp-gpu
```

2. Запуск с автозагрузкой после перезагрузки ПК

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now llama-cpp # или llama-cpp-gpu
```

## Настройка локального ПК для запуска [opencode](https://github.com/anomalyco/opencode/blob/dev/README.ru.md)

Любой ПК -- в соответствии с [документацией](https://github.com/anomalyco/opencode/blob/dev/README.ru.md)

Тестирование проводилось на ПК:
- Host ОС -- Windows 11 Home -> WSL: KaliLinux:2026.1
- Intel Ultra 125H
- 32 Gb RAM

### Установка opencode на Локальный ПК:

```bash
curl -fsSL https://opencode.ai/install | bash
```

```bash
git clone https://github.com/danil1online/PyRuQo1.git
cd PyRuQo1
```

Редактирование файла настройки доступа

- `opencode.json` размещается в локальной папке и действует только в ее пределах
- `~/.config/opencode/opencode.json` -- конфиг для учетной записи пользователя

1. Глобальная настройка -- для всех проектов
```bash
mkdir ~/.config/opencode
nano ~/.config/opencode/opencode.json
```

2. Можно настроить для отдельного проекта, для этого в папке проекта создать и заполнить
```bash
nano opencode.json
```

Пример самого подробного заполнения `opencode.json` (или `~/.config/opencode/opencode.json`).

Пример обеспечивает:
- подключение к модели на llm-сервере, через промежуточный VPS|VDS `"baseURL": "http://195.63.13.56:8080/v1"`
- `"attachment": true,` позволяет отправлять модели документы, если включен мультимодальный режим - скриншоты
- `"input": ["text", "image"],` включает мультимодальный режим входных данных, `"output": ["text"]` - выход "только текст"
- параметры `"limit": {"context": 196608,"output": 16384}` делают возможным наблюдение за расходом контекста, иначе opencode показыват всегда 0 % (правая часть пользовательского интерфейса, показывается если растянуть окно)
- `"tools"` и `"permission"` - позволяют модели искать информацию в интеренете, работать с диском локального ПК без дополнительных вопросов.

**Обратить внимание** название `unsloth/qwen3.6-35b-a3b` не критично для llama.cpp, но если LLM-сервер поднят на lm-studio, знать точное название обязательно. Модели размещаются при этом на llm-сервере в `/home/user/.lmstudio/models/unsloth/` 
- 


```sh
{
  "$schema": "https://opencode.ai/config.json",
  "model": "openai-compatible/qwen-local",
  "provider": {
    "openai-compatible": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Llama.cpp",
      "options": {
        "baseURL": "http://195.63.13.56:8080/v1"
      },
      "models": {
        "qwen-local": {
          "id": "unsloth/qwen3.6-35b-a3b",
          "attachment": true,
          "modalities": {
            "input": ["text", "image"],
            "output": ["text"]
          },
          "limit": {
            "context": 196608,
            "output": 16384
          }
        }
      }
    }
  },
  "tools": {
    "websearch": true,
    "webfetch": true,
    "codesearch": true
  },
  "permission": {
    "websearch": "allow",
    "webfetch": "allow",
    "codesearch": "allow",
    "read": "allow",
    "edit": "allow",
    "bash": "allow",
    "external_directory": "allow"
  }
}
```


Запуск из анализируемого каталога:

```bash
opencode
```
или (при необходимости восстановления контекста, даже после перезагрузки)
```bash
opencode -c
```
или
```bash
opencode -c -m openai-compatible/qwen-local
```
-> Tab для перехода в режим планирования / анализа Plan ("стартовый" Build будет сразу править код)
-> Пример запроса: "Какие улучшения для проекта можешь предложить?"

# Доступ к llama.cpp | lmstuido через сеть интернет. VPS|VDS

Минимальный из доступных на рынке. ОС -- Linux, т.е. с возможностью управлять маршрутизацией через iptables

Например:

RuVDS
- Linxdatacenter: Россия, Санкт-Петербург
- Ubuntu 22.04 LTS (ENG)
- 1x2.2ГГц, 0.5Гб RAM, 1IP
- 10Гб HDD RAID (Операционная система)


Установлен [openvpn-сервер](https://habr.com/ru/articles/912336/)

Прописаны настройки доступа к [серверу LLM](#сервер-llm):

Для [llama-cpp](#установка-и-запуск-llama.cpp), VDS с IP 195.66.13.56, LLM-сервера с ip внутри openvpn-сети 10.8.0.7
```bash
iptables -t nat -A PREROUTING -d 195.66.13.56 -p tcp --dport 8080 -j DNAT --to-destination 10.8.0.7
```

Для [lm-studio](#настройки-lm-studio), VDS с IP 195.66.13.56, LLM-сервера с ip внутри openvpn-сети 10.8.0.7
```bash
iptables -t nat -A PREROUTING -d 195.66.13.56 -p tcp --dport 1234 -j DNAT --to-destination 10.8.0.7
```

Сохранение:
```bash
netfilter-persistent save
```

Дополнительные команды:
```bash
iptables -t nat -v -L PREROUTING -n --line-number # список настроенных правил
iptables -t nat -D PREROUTING 13                  # удаление правила за номером 13
```

# Настройки LM-Studio:

![Подключение и настройка модели](Images/part1.jpg)

![Дополнительная настройка: "Number of layers for MoE onto CPU 40"](Images/part2.jpg)

# Список использованных источников

- [Базовая статья](https://habr.com/ru/articles/1026482/) 
- [Qwen3.6 35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main).
- [opencode](https://github.com/anomalyco/opencode/blob/dev/README.ru.md)