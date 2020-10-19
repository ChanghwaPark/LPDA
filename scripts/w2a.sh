#!/bin/bash

src='webcam'
trg='amazon'
gpu=${1}

python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0 --lw 0.01 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0 --lw 0.1 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0 --lw 1.0 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0.01 --lw 0 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0.01 --lw 0.01 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0.01 --lw 0.1 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0.01 --lw 1.0 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0.1 --lw 0 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0.1 --lw 0.01 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0.1 --lw 0.1 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 0.1 --lw 1.0 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 1.0 --lw 0 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 1.0 --lw 0.01 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 1.0 --lw 0.1 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
python main.py --src ${src} --trg ${trg} --bs 36 --nn resnet \
--dw 1.0 --lw 1.0 --cgw 0 --xw 0 --sw 0 --tw 0 --sv 0 --tw 0 --gpu ${gpu}
